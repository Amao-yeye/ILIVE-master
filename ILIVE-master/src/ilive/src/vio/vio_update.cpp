#include "ilive.hpp"
#include "Basis/tools/tools_mem_used.h"
#include "Basis/tools/tools_logger.hpp"
#define USING_CERES 0


double get_huber_loss_scale(double reprojection_error, double outlier_threshold = 1.0)
{
    double scale = 1.0;
    if (reprojection_error / outlier_threshold < 1.0)
    {
        scale = 1.0;
    }
    else
    {
        scale = (2 * sqrt(reprojection_error) / sqrt(outlier_threshold) - 1.0) / reprojection_error;
    }
    return scale;
}


const int minimum_iteration_pts = 10;
bool ILIVE::vio_projection( StatesGroup &state_in, Rgbmap_tracker &op_track )
{
    Common_tools::Timer tim;
    tim.tic();
    scope_color( ANSI_COLOR_BLUE_BOLD );
    StatesGroup state_iter = state_in;
    if ( !m_if_estimate_intrinsic ) // When disable the online intrinsic calibration.
    {
        state_iter.cam_intrinsic << g_cam_K( 0, 0 ), g_cam_K( 1, 1 ), g_cam_K( 0, 2 ), g_cam_K( 1, 2 );
    }

    if ( !m_if_estimate_i2c_extrinsic )
    {
        state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;
        state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;
    }

    Eigen::Matrix< double, -1, -1 >                       H_mat;
    Eigen::Matrix< double, -1, 1 >                        meas_vec;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    Eigen::Matrix< double, DIM_OF_STATES, 1 >             solution;
    Eigen::Matrix< double, -1, -1 >                       K, KH;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > K_1;

    Eigen::SparseMatrix< double > H_mat_spa, H_T_H_spa, K_spa, KH_spa, vec_spa, I_STATE_spa;
    I_STATE.setIdentity();
    I_STATE_spa = I_STATE.sparseView();
    double fx, fy, cx, cy, time_td;

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size );

    if ( total_pt_size < minimum_iteration_pts )
    {
        state_in = state_iter;
        return false;
    }
    H_mat.resize( total_pt_size * 2, DIM_OF_STATES );
    meas_vec.resize( total_pt_size * 2, 1 );
    double last_repro_err = 3e8;
    int    avail_pt_count = 0;
    double last_avr_repro_err = 0;

    double acc_reprojection_error = 0;
    double img_res_scale = 1.0;
    for ( int iter_count = 0; iter_count < esikf_iter_times; iter_count++ )
    {

        // cout << "========== Iter " << iter_count << " =========" << endl;
        mat_3_3 R_imu = state_iter.rot_end;
        vec_3   t_imu = state_iter.pos_end;
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu;
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // world to camera frame

        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta;

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w;
        mat_3_3 R_w2c = R_c2w.transpose();
        int     pt_idx = -1;
        acc_reprojection_error = 0;
        vec_3               pt_3d_w, pt_3d_cam;
        vec_2               pt_img_measure, pt_img_proj, pt_img_vel;
        eigen_mat_d< 2, 3 > mat_pre;
        eigen_mat_d< 3, 3 > mat_A, mat_B, mat_C, mat_D, pt_hat;
        H_mat.setZero();
        solution.setZero();
        meas_vec.setZero();
        avail_pt_count = 0;
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;
            pt_img_measure = vec_2( it->second.x, it->second.y );
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c;
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;
            double repro_err = ( pt_img_proj - pt_img_measure ).norm();
            double huber_loss_scale = get_huber_loss_scale( repro_err );
            pt_idx++;
            acc_reprojection_error += repro_err;
            // if (iter_count == 0 || ((repro_err - last_reprojection_error_vec[pt_idx]) < 1.5))
            if ( iter_count == 0 || ( ( repro_err - last_avr_repro_err * 5.0 ) < 0 ) )
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            else
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            avail_pt_count++;
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );

            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );
            mat_C = Sophus::SO3d::hat( pt_3d_cam );
            mat_D = -state_iter.rot_ext_i2c.transpose();
            meas_vec.block( pt_idx * 2, 0, 2, 1 ) = ( pt_img_proj - pt_img_measure ) * huber_loss_scale / img_res_scale;

            H_mat.block( pt_idx * 2, 0, 2, 3 ) = mat_pre * mat_A * huber_loss_scale;
            H_mat.block( pt_idx * 2, 3, 2, 3 ) = mat_pre * mat_B * huber_loss_scale;
            if ( DIM_OF_STATES > 24 )
            {
                H_mat.block( pt_idx * 2, 24, 2, 1 ) = pt_img_vel * huber_loss_scale;
            }
            if ( m_if_estimate_i2c_extrinsic )
            {
                H_mat.block( pt_idx * 2, 18, 2, 3 ) = mat_pre * mat_C * huber_loss_scale;
                H_mat.block( pt_idx * 2, 21, 2, 3 ) = mat_pre * mat_D * huber_loss_scale;
            }

            if ( m_if_estimate_intrinsic )
            {
                H_mat( pt_idx * 2, 25 ) = pt_3d_cam( 0 ) / pt_3d_cam( 2 ) * huber_loss_scale;
                H_mat( pt_idx * 2 + 1, 26 ) = pt_3d_cam( 1 ) / pt_3d_cam( 2 ) * huber_loss_scale;
                H_mat( pt_idx * 2, 27 ) = 1 * huber_loss_scale;
                H_mat( pt_idx * 2 + 1, 28 ) = 1 * huber_loss_scale;
            }
        }
        H_mat = H_mat / img_res_scale;
        acc_reprojection_error /= total_pt_size;

        last_avr_repro_err = acc_reprojection_error;
        if ( avail_pt_count < minimum_iteration_pts )
        {
            break;
        }

        H_mat_spa = H_mat.sparseView();
        Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose();
        vec_spa = ( state_iter - state_in ).sparseView();
        H_T_H_spa = Hsub_T_temp_mat * H_mat_spa;
        // Notice that we have combine some matrix using () in order to boost the matrix multiplication.
        Eigen::SparseMatrix< double > temp_inv_mat =
            ( ( H_T_H_spa.toDense() + eigen_mat< -1, -1 >( state_in.cov * m_cam_measurement_weight ).inverse() ).inverse() ).sparseView();
        KH_spa = temp_inv_mat * ( Hsub_T_temp_mat * H_mat_spa );
        solution = ( temp_inv_mat * ( Hsub_T_temp_mat * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - KH_spa ) * vec_spa ).toDense();

        state_iter = state_iter + solution;

        if ( fabs( acc_reprojection_error - last_repro_err ) < 0.01 )
        {
            break;
        }
        last_repro_err = acc_reprojection_error;
    }

    if ( avail_pt_count >= minimum_iteration_pts )
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();
    }

    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;
    state_iter.td_ext_i2c_delta = 0;
    state_in = state_iter;
    return true;
}

bool ILIVE::vio_photometric( StatesGroup &state_in, Rgbmap_tracker &op_track, std::shared_ptr< Image_frame > &image )
{
    Common_tools::Timer tim;
    tim.tic();
    StatesGroup state_iter = state_in;
    if ( !m_if_estimate_intrinsic )     // When disable the online intrinsic calibration.
    {
        state_iter.cam_intrinsic << g_cam_K( 0, 0 ), g_cam_K( 1, 1 ), g_cam_K( 0, 2 ), g_cam_K( 1, 2 );
    }
    if ( !m_if_estimate_i2c_extrinsic ) // When disable the online extrinsic calibration.
    {
        state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;
        state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;
    }
    Eigen::Matrix< double, -1, -1 >                       H_mat, R_mat_inv;
    Eigen::Matrix< double, -1, 1 >                        meas_vec;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    Eigen::Matrix< double, DIM_OF_STATES, 1 >             solution;
    Eigen::Matrix< double, -1, -1 >                       K, KH;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > K_1;
    Eigen::SparseMatrix< double >                         H_mat_spa, H_T_H_spa, R_mat_inv_spa, K_spa, KH_spa, vec_spa, I_STATE_spa;
    I_STATE.setIdentity();
    I_STATE_spa = I_STATE.sparseView();
    double fx, fy, cx, cy, time_td;

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size );
    if ( total_pt_size < minimum_iteration_pts )
    {
        state_in = state_iter;
        return false;
    }

    int err_size = 3;
    H_mat.resize( total_pt_size * err_size, DIM_OF_STATES );
    meas_vec.resize( total_pt_size * err_size, 1 );
    R_mat_inv.resize( total_pt_size * err_size, total_pt_size * err_size );

    double last_repro_err = 3e8;
    int    avail_pt_count = 0;
    double last_avr_repro_err = 0;
    int    if_esikf = 1;

    double acc_photometric_error = 0;
    for ( int iter_count = 0; iter_count < 2; iter_count++ )
    {
        mat_3_3 R_imu = state_iter.rot_end;
        vec_3   t_imu = state_iter.pos_end;
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu;
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // world to camera frame

        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta;

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w;
        mat_3_3 R_w2c = R_c2w.transpose();
        int     pt_idx = -1;
        acc_photometric_error = 0;
        vec_3               pt_3d_w, pt_3d_cam;
        vec_2               pt_img_measure, pt_img_proj, pt_img_vel;
        eigen_mat_d< 2, 3 > mat_pre;
        eigen_mat_d< 3, 2 > mat_photometric;
        eigen_mat_d< 3, 3 > mat_d_pho_d_img;
        eigen_mat_d< 3, 3 > mat_A, mat_B, mat_C, mat_D, pt_hat;
        R_mat_inv.setZero();
        H_mat.setZero();
        solution.setZero();
        meas_vec.setZero();
        avail_pt_count = 0;
        int iter_layer = 0;
        tim.tic( "Build_cost" );
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            if ( ( ( RGB_pts * ) it->first )->m_N_rgb < 3 )
            {
                continue;
            }
            pt_idx++;
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;
            pt_img_measure = vec_2( it->second.x, it->second.y );
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c;
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;

            vec_3   pt_rgb = ( ( RGB_pts * ) it->first )->get_rgb();
            mat_3_3 pt_rgb_info = mat_3_3::Zero();
            mat_3_3 pt_rgb_cov = ( ( RGB_pts * ) it->first )->get_rgb_cov();
            for ( int i = 0; i < 3; i++ )
            {
                pt_rgb_info( i, i ) = 1.0 / pt_rgb_cov( i, i ) / 255;
                R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) = pt_rgb_info( i, i );
                // R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) =  1.0;
            }
            vec_3  obs_rgb_dx, obs_rgb_dy;
            vec_3  obs_rgb = image->get_rgb( pt_img_proj( 0 ), pt_img_proj( 1 ), 0, &obs_rgb_dx, &obs_rgb_dy );
            vec_3  photometric_err_vec = ( obs_rgb - pt_rgb );
            double huber_loss_scale = get_huber_loss_scale( photometric_err_vec.norm() );
            photometric_err_vec *= huber_loss_scale;
            double photometric_err = photometric_err_vec.transpose() * pt_rgb_info * photometric_err_vec;

            acc_photometric_error += photometric_err;

            last_reprojection_error_vec[ pt_idx ] = photometric_err;

            avail_pt_count++;
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );
            mat_d_pho_d_img = mat_photometric * mat_pre;

            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );
            mat_C = Sophus::SO3d::hat( pt_3d_cam );
            mat_D = -state_iter.rot_ext_i2c.transpose();
            meas_vec.block( pt_idx * 3, 0, 3, 1 ) = photometric_err_vec ;

            H_mat.block( pt_idx * 3, 0, 3, 3 ) = mat_d_pho_d_img * mat_A * huber_loss_scale;
            H_mat.block( pt_idx * 3, 3, 3, 3 ) = mat_d_pho_d_img * mat_B * huber_loss_scale;
            if ( 1 )
            {
                if ( m_if_estimate_i2c_extrinsic )
                {
                    H_mat.block( pt_idx * 3, 18, 3, 3 ) = mat_d_pho_d_img * mat_C * huber_loss_scale;
                    H_mat.block( pt_idx * 3, 21, 3, 3 ) = mat_d_pho_d_img * mat_D * huber_loss_scale;
                }
            }
        }
        R_mat_inv_spa = R_mat_inv.sparseView();
       
        last_avr_repro_err = acc_photometric_error;
        if ( avail_pt_count < minimum_iteration_pts )
        {
            break;
        }
        // Esikf
        tim.tic( "Iter" );
        if ( if_esikf )
        {
            H_mat_spa = H_mat.sparseView();
            Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose();
            vec_spa = ( state_iter - state_in ).sparseView();
            H_T_H_spa = Hsub_T_temp_mat * R_mat_inv_spa * H_mat_spa;
            Eigen::SparseMatrix< double > temp_inv_mat =
                ( H_T_H_spa.toDense() + ( state_in.cov * m_cam_measurement_weight ).inverse() ).inverse().sparseView();
            Eigen::SparseMatrix< double > Ht_R_inv = ( Hsub_T_temp_mat * R_mat_inv_spa );
            KH_spa = Ht_R_inv * H_mat_spa;
            solution =
                ( temp_inv_mat * ( Ht_R_inv * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - temp_inv_mat * KH_spa ) * vec_spa ).toDense();
        }
        state_iter = state_iter + solution;

        if ( ( acc_photometric_error / total_pt_size ) < 10 ) // By experience.
        {
            break;
        }
        if ( fabs( acc_photometric_error - last_repro_err ) < 0.01 )
        {
            break;
        }
        last_repro_err = acc_photometric_error;
    }
    if ( if_esikf && avail_pt_count >= minimum_iteration_pts )
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();
    }
    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;
    state_iter.td_ext_i2c_delta = 0;
    state_in = state_iter;
    return true;
}
