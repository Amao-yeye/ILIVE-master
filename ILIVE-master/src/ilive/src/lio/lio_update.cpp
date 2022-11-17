#include "ilive.hpp"


void ILIVE::lio_update(Common_tools::Timer tim, bool rematch_en, StatesGroup state_propagate, PointCloudXYZINormal::Ptr laserCloudOri, PointCloudXYZINormal::Ptr coeffSel,
    PointCloudXYZINormal::Ptr feats_down, PointCloudXYZINormal::Ptr feats_down_updated, PointCloudXYZINormal::Ptr coeffSel_tmpt, int feats_down_size, int  rematch_num,
    double maximum_pt_range, double match_time, double solve_time, double match_start, double pca_time, double solve_start, std::vector< PointVector > Nearest_Points, 
    std::vector< bool >  point_selected_surf, std::vector< double >     res_last, Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >  H_T_H,
    Eigen::Vector3d euler_cur, Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > I_STATE){
   for ( iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ ) // NUM_MAX_ITERATIONS默认为4
    {
        tim.tic( "Iter" );  // 本次迭代起始时间
        match_start = omp_get_wtime();
        laserCloudOri->clear(); // 清空存放找到了最近平面的点的容器
        coeffSel->clear();      // 清空存放最近平面信息的容器

        for ( int i = 0; i < feats_down_size; i += m_lio_update_point_step ) // m_lio_update_point_step默认为1
        {
            double     search_start = omp_get_wtime();
            PointType &pointOri_tmpt = feats_down->points[ i ]; // 获取当前下标的特征点 - 原始点
            // 计算特征点与原点的距离
            double   ori_pt_dis = sqrt( pointOri_tmpt.x * pointOri_tmpt.x + pointOri_tmpt.y * pointOri_tmpt.y + pointOri_tmpt.z * pointOri_tmpt.z );
            maximum_pt_range = std::max( ori_pt_dis,   maximum_pt_range );// 保存离原点最远的点产生的最远距离
            PointType &pointSel_tmpt =   feats_down_updated->points[ i ]; // 获取当前下标的特征点 - 更新后的点

            /* transform to world frame */
            pointBodyToWorld( &pointOri_tmpt, &pointSel_tmpt );// 将特征点转到世界坐标下,并保存至可变点pointSel_tmpt中
            std::vector< float > pointSearchSqDis_surf; // 搜索点-平面时产生的距离序列

            auto &points_near =   Nearest_Points[ i ];    // 下标i的特征点的最近点集合(一维数组)


            if ( iterCount == 0 ||   rematch_en ) // 第一次迭代或者重匹配使能时才在ikd树上搜索最近平面
            {
                  point_selected_surf[ i ] = true;
                /** Find the closest surfaces in the map 在地图中找到最近的平面**/
                // NUM_MATCH_POINTS=5
                ikdtree.Nearest_Search( pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf );
                float max_distance = pointSearchSqDis_surf[ NUM_MATCH_POINTS - 1 ];//最近点集的最后一个元素自然最远
                if ( max_distance > m_maximum_pt_kdtree_dis )   // 超出限定距离,放弃为这个点寻找平面
                {
                      point_selected_surf[ i ] = false;   // 当前点寻找平面失败
                }
            }

            kdtree_search_time += omp_get_wtime() - search_start;
            if (   point_selected_surf[ i ] == false )    // 当前点寻找平面失败,进入下一个点的寻找流程
                continue;

            double pca_start = omp_get_wtime();

            cv::Mat matA0( NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all( 0 ) );
            cv::Mat matB0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( -1 ) );
            cv::Mat matX0( NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all( 0 ) );

            for ( int j = 0; j < NUM_MATCH_POINTS; j++ )
            {
                matA0.at< float >( j, 0 ) = points_near[ j ].x;
                matA0.at< float >( j, 1 ) = points_near[ j ].y;
                matA0.at< float >( j, 2 ) = points_near[ j ].z;
            }

            cv::solve( matA0, matB0, matX0, cv::DECOMP_QR ); // TODO

            float pa = matX0.at< float >( 0, 0 );
            float pb = matX0.at< float >( 1, 0 );
            float pc = matX0.at< float >( 2, 0 );
            float pd = 1;

            float ps = sqrt( pa * pa + pb * pb + pc * pc );
            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            bool planeValid = true;
            for ( int j = 0; j < NUM_MATCH_POINTS; j++ )    
            {
                if ( fabs( pa * points_near[ j ].x + pb * points_near[ j ].y + pc * points_near[ j ].z + pd ) >
                        m_planar_check_dis ) // Raw 0.05
                {
                    if ( ori_pt_dis <   maximum_pt_range * 0.90 || ( ori_pt_dis < m_long_rang_pt_dis ) )
                    // if(1)
                    {
                        planeValid = false;
                          point_selected_surf[ i ] = false;  
                        break;
                    }
                }
            }

            if ( planeValid )
            {
 
                float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                float s = 1 - 0.9 * fabs( pd2 ) /
                                    sqrt( sqrt( pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y +
                                                pointSel_tmpt.z * pointSel_tmpt.z ) );
                double acc_distance = ( ori_pt_dis < m_long_rang_pt_dis ) ? m_maximum_res_dis : 1.0;
                if ( pd2 < acc_distance )   // 残差小于0.3或者1.0
                {
                      point_selected_surf[ i ] = true;    // 当前点寻找平面成功
                      coeffSel_tmpt->points[ i ].x = pa;  // 记录当前特征点对应最近平面的平面方程
                      coeffSel_tmpt->points[ i ].y = pb;
                      coeffSel_tmpt->points[ i ].z = pc;
                      coeffSel_tmpt->points[ i ].intensity = pd2;
                      res_last[ i ] = std::abs( pd2 );    // 当前特征点代入平面方程产生的残差
                }
                else
                {
                      point_selected_surf[ i ] = false;// 当前点寻找平面失败
                }
            }
              pca_time += omp_get_wtime() - pca_start;
        }  
        tim.tic( "Stack" );
        double total_residual = 0.0;
        laserCloudSelNum = 0;

        for ( int i = 0; i <   coeffSel_tmpt->points.size(); i++ )    // 遍历找到了对应平面的特征点
        {
            if (   point_selected_surf[ i ] && (   res_last[ i ] <= 2.0 ) )
            {                                                           // 下面重新放入容器是为了对齐点-面的索引
                laserCloudOri->push_back(   feats_down->points[ i ] );    // 将找到最近平面的点放入laserCloudOri中
                  coeffSel->push_back(   coeffSel_tmpt->points[ i ] );      // 将最近平面容器放入coeffSel中
                total_residual +=   res_last[ i ];        // 总残差 - 从最小二乘的角度,优化的就是让这个总残差最小
                laserCloudSelNum++;                     // 找到平面的点的数量
            }
        }
        res_mean_last = total_residual / laserCloudSelNum;  // 均值-期望, 后面没用此变量

        match_time += omp_get_wtime() - match_start;
        solve_start = omp_get_wtime();

        Eigen::MatrixXd Hsub( laserCloudSelNum, 6 );    // Hsub(n x 6)
       
        Eigen::VectorXd meas_vec( laserCloudSelNum );   // meas_vec(n x 1)
        Hsub.setZero();
        for ( int i = 0; i < laserCloudSelNum; i++ )
        {
            const PointType &laser_p =   laserCloudOri->points[ i ];// 获取当前点
            Eigen::Vector3d  point_this( laser_p.x, laser_p.y, laser_p.z );// 点坐标
            point_this += ext_t_lid_in_imu;  // Lidar和IMU的偏移
            Eigen::Matrix3d point_crossmat;
            point_crossmat << SKEW_SYM_MATRIX( point_this );    // 将点转为反对称矩阵用于叉乘

            const PointType &norm_p = coeffSel->points[ i ];    // 当前点的最近平面方程系数
            Eigen::Vector3d  norm_vec( norm_p.x, norm_p.y, norm_p.z );// 平面法向量
            Eigen::Vector3d A( point_crossmat * g_lio_state.rot_end.transpose() * norm_vec );
            Hsub.row( i ) << VEC_FROM_ARRAY( A ), norm_p.x, norm_p.y, norm_p.z;// row(i)=A[0],A[1],A[2],norm_p.x, norm_p.y, norm_p.z

            /*** Measuremnt: distance to the closest surface/corner ***/
            meas_vec( i ) = -norm_p.intensity;  
        }

        Eigen::Vector3d                           rot_add, t_add, v_add, bg_add, ba_add, g_add; //更新量:旋转,平移,速度,偏置等
        Eigen::Matrix< double, DIM_OF_STATES, 1 > solution; // 最终解 : 29维
        Eigen::MatrixXd                           K( DIM_OF_STATES, laserCloudSelNum );// kalman增益


        if ( !flg_EKF_inited )  // 未初始化时初始化 - 前面已经初始化了
        {
            cout << ANSI_COLOR_RED_BOLD << "Run EKF init" << ANSI_COLOR_RESET << endl;
            /*** only run in initialization period ***/
            set_initial_state_cov( g_lio_state );
        } 
        else{
            auto && Hsub_T = Hsub.transpose();   // H转置 : 6xn = (nx6)^T
              H_T_H.block< 6, 6 >( 0, 0 ) = Hsub_T * Hsub;//(0,0)处6x6块.H^T*T
            Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > &&K_1 =
                (   H_T_H + ( g_lio_state.cov / LASER_POINT_COV ).inverse() ).inverse();
            K = K_1.block< DIM_OF_STATES, 6 >( 0, 0 ) * Hsub_T;// K = (29x6) * (6xn) = (29xn)


            auto vec =   state_propagate - g_lio_state;//state_propagate初始=g_lio_state
            //5>:求公式(18)的中间和右边部分(有出入:I什么的都省略了)
            solution = K * ( meas_vec - Hsub * vec.block< 6, 1 >( 0, 0 ) ); // kalman增益
            g_lio_state = state_propagate + solution;   // kalman增益后的状态结果
            print_dash_board();
            // cout << ANSI_COLOR_RED_BOLD << "Run EKF uph, vec = " << vec.head<9>().transpose() << ANSI_COLOR_RESET << endl;
            rot_add = solution.block< 3, 1 >( 0, 0 );   // 旋转增量
            t_add = solution.block< 3, 1 >( 3, 0 );     // 平移增量
            flg_EKF_converged = false;                  // 收敛标识
            //7>:判断是否收敛
            if ( ( ( rot_add.norm() * 57.3 - deltaR ) < 0.01 ) && ( ( t_add.norm() * 100 - deltaT ) < 0.015 ) )
            {
                flg_EKF_converged = true;   // 通过旋转和平移增量与上一次迭代的差值,判断是否收敛
                                            // ? : 收敛了为啥不加break,而是继续进行迭代
            }
            //8>:旋转和平移增量转换单位
            deltaR = rot_add.norm() * 57.3; // 角度单位
            deltaT = t_add.norm() * 100;    // 厘米单位
        }

        // printf_line;
        g_lio_state.last_update_time = Measures.lidar_end_time;
          euler_cur = RotMtoEuler( g_lio_state.rot_end ); // 获得当前lidar的里程计信息,最后这个需要发布到ros中去
        dump_lio_state_to_log( m_lio_state_fp );

        /*** Rematch Judgement 重匹配判断 ***/
        rematch_en = false;
        if ( flg_EKF_converged || ( (   rematch_num == 0 ) && ( iterCount == ( NUM_MAX_ITERATIONS - 2 ) ) ) )
        {
              rematch_en = true;
              rematch_num++;
        }

        if (   rematch_num >= 2 || ( iterCount == NUM_MAX_ITERATIONS - 1 ) ) 
        {
            if ( flg_EKF_inited )
            {
                G.block< DIM_OF_STATES, 6 >( 0, 0 ) = K * Hsub; // 对应公式(19)中 : K * H
                g_lio_state.cov = (   I_STATE -   G ) * g_lio_state.cov;//公式(19): (单位阵-K*H)*Cur_协方差
                total_distance += ( g_lio_state.pos_end - position_last ).norm();// 两次state间的距离
                position_last = g_lio_state.pos_end;   
            }
              solve_time += omp_get_wtime() -   solve_start;
            break;
        }
          solve_time += omp_get_wtime() -   solve_start;
    }
    return;
}

