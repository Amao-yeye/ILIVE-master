#include "ilive.hpp"


/**
 * @note LIO子系统处理线程
 */ 
int ILIVE::service_LIO_update()
{
    
    //--------Variable definition and initialization-----------------------------------
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    PointCloudXYZINormal::Ptr feats_undistort( new PointCloudXYZINormal() );//去除畸变后的点云
    PointCloudXYZINormal::Ptr feats_down( new PointCloudXYZINormal() );     // 保存下采样后的点云

    PointCloudXYZINormal::Ptr coeffSel( new PointCloudXYZINormal() );// 存放M个最近平面信息的容器: 平面方程,点-面残差
    PointCloudXYZINormal::Ptr laserCloudOri( new PointCloudXYZINormal() );// 存放找到了最近平面的M个点的容器


    nav_msgs::Path path;    // Lidar的路径 : 主要有两个成员变量: header和pose
    path.header.stamp = ros::Time::now();   // header的时间
    path.header.frame_id = "/world";        // header的id
    cv::Mat matA1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );    // 后面没用到
    cv::Mat matD1( 1, 3, CV_32F, cv::Scalar::all( 0 ) );    // 后面没用到
    cv::Mat matV1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );    // 后面没用到
    cv::Mat matP( 6, 6, CV_32F, cv::Scalar::all( 0 ) );    // 后面没用到

    /*** variables initialize ***/
    FOV_DEG = fov_deg + 10; // fov_deg=360
    HALF_FOV_COS = std::cos( ( fov_deg + 10.0 ) * 0.5 * PI_M / 180.0 );// cos(185)

    for ( int i = 0; i < laserCloudNum; i++ )   // laserCloudNum = 48x48x48
    {
        featsArray[ i ].reset( new PointCloudXYZINormal() );
    }

    std::shared_ptr< ImuProcess > p_imu( new ImuProcess() );    // 定义用于前向/后向传播的IMU处理器
    m_imu_process = p_imu;
    //------------------------------------------------------------------------------------------------------
    ros::Rate rate( 5000 );
    bool status = ros::ok();
    g_camera_lidar_queue.m_liar_frame_buf = &lidar_buffer;//取出lidar数据
    set_initial_state_cov( g_lio_state );   //初始化g_lio_state的状态协方差矩阵

    while ( ros::ok() ) //运行LIO线程循环
    {
        
        if ( flg_exit ) break;
        ros::spinOnce();
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        while ( g_camera_lidar_queue.if_lidar_can_process() == false )
        {   
            ros::spinOnce();
            std::this_thread::yield();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
        }
        std::unique_lock< std::mutex > lock( m_mutex_lio_process ); 
        if ( 1 )
        {
            // printf_line;
            Common_tools::Timer tim;

            if ( sync_packages( Measures ) == 0 )
            {
                continue;   // 提取数据失败
            }
            int lidar_can_update = 1;   
            g_lidar_star_tim = frame_first_pt_time;
            if ( flg_reset )    // 判断重置标识
            {
                ROS_WARN( "reset when rosbag play back" );
                p_imu->Reset(); // 重置前向/后向传播用的处理器 : 重置处理时用到的期望和方差等变量
                flg_reset = false;
                continue;
            }
            g_LiDAR_frame_index++;      // lidar帧++
            tim.tic( "Preprocess" );    // time_current : 获取当前时间
            double t0, t1, t2, t3, t4, t5, match_start, match_time, solve_start, solve_time, pca_time, svd_time;
            // 重置处理时用于记录各块处理时间的变量
            match_time = 0;
            kdtree_search_time = 0;
            solve_time = 0;
            pca_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            p_imu->Process( Measures, g_lio_state, feats_undistort );

            g_camera_lidar_queue.g_noise_cov_acc = p_imu->cov_acc;  // 获取加速度误差状态传递的协方差
            g_camera_lidar_queue.g_noise_cov_gyro = p_imu->cov_gyr; // 获取角速度误差状态传递的协方差
            StatesGroup state_propagate( g_lio_state ); // 状态传播值(先验):通过计算得到的状态实例化一个StatesGroup变量

            // 输出lio上一帧更新的时间 : 上一帧更新记录时间 - lidar开始时间
            // cout << "G_lio_state.last_update_time =  " << std::setprecision(10) << g_lio_state.last_update_time -g_lidar_star_tim  << endl;
            if ( feats_undistort->empty() || ( feats_undistort == NULL ) )  // 没有成功去除点云运动畸变
            {

                frame_first_pt_time = Measures.lidar_beg_time;
                std::cout << "not ready for odometry" << std::endl;
                continue;
            }

            if ( ( Measures.lidar_beg_time - frame_first_pt_time ) < INIT_TIME ) // INIT_TIME=0
            {
                flg_EKF_inited = false;
                std::cout << "||||||||||Initiallizing LiDAR||||||||||" << std::endl;
            }
            else    // 时间满足关系,开始EKF过程
            {
                flg_EKF_inited = true;
            }
            /*** Compute the euler angle 这里的euler_cur就是当前的lidar里程计的旋转信息,后面需要用kalman迭代更新,最后发布到ros中***/
            Eigen::Vector3d euler_cur = RotMtoEuler( g_lio_state.rot_end );// 最后时刻时lidar的旋转向量 : 四元数


#ifdef DEBUG_PRINT  // 默认注释了DEBUG_PRINT的定义
            std::cout << "current lidar time " << Measures.lidar_beg_time << " "
                      << "first lidar time " << frame_first_pt_time << std::endl;
            // 打印预积分后的结果(最后时刻IMU的状态) : 旋转向量(1rad=57.3度), 位置向量, 速度向量, 角速度bias向量, 加速度bias量
            std::cout << "pre-integrated states: " << euler_cur.transpose() * 57.3 << " " << g_lio_state.pos_end.transpose() << " "
                      << g_lio_state.vel_end.transpose() << " " << g_lio_state.bias_g.transpose() << " " << g_lio_state.bias_a.transpose()
                      << std::endl;
#endif

            lasermap_fov_segment();//为防地图大小不受约束,ikd树保留lidar位置周围长度为L的局部区域地图点.

            downSizeFilterSurf.setInputCloud( feats_undistort );//构建三维体素栅格 
            downSizeFilterSurf.filter( *feats_down );           //下采样滤波

            if ( ( feats_down->points.size() > 1 ) && ( ikdtree.Root_Node == nullptr ) )
            {
                // std::vector<PointType> points_init = feats_down->points;
                ikdtree.set_downsample_param( filter_size_map_min ); // filter_size_map_min默认=0.4
                ikdtree.Build( feats_down->points );    // 构造idk树
                flg_map_initialized = true;
                continue;   // 进入下一次循环
            }

            if ( ikdtree.Root_Node == nullptr ) // 构造ikd树失败
            {
                flg_map_initialized = false;
                std::cout << "~~~~~~~ Initialize Map iKD-Tree Failed! ~~~~~~~" << std::endl;
                continue;
            }
            int featsFromMapNum = ikdtree.size();   // ikd树的节点数
            int feats_down_size = feats_down->points.size();    // 下采样过滤后的点数

            PointCloudXYZINormal::Ptr coeffSel_tmpt( new PointCloudXYZINormal( *feats_down ) );
            PointCloudXYZINormal::Ptr feats_down_updated( new PointCloudXYZINormal( *feats_down ) );
 
          
            std::vector< double >     res_last( feats_down_size, 1000.0 ); // initial : 存放每个特征点的残差值
            if ( featsFromMapNum >= 5 ) // ***重点*** : 正式开始ICP和迭代Kalman : ikd树上至少有5个点才进行操作
            {
                t1 = omp_get_wtime();

                /**
                 * @note (2-7-1) : 在ros上发布特征点云数据 - 默认不发布
                 */ 
                if ( m_if_publish_feature_map ) 
                {
                    PointVector().swap( ikdtree.PCL_Storage );
                    // flatten会将需要删除的点放入Points_deleted或Multithread_Points_deleted中
                    ikdtree.flatten( ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD );
                    featsFromMap->clear();
                    featsFromMap->points = ikdtree.PCL_Storage;

                    sensor_msgs::PointCloud2 laserCloudMap;
                    pcl::toROSMsg( *featsFromMap, laserCloudMap );  // 将点云数据格式转换为发布的消息格式
                    laserCloudMap.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
                    // laserCloudMap.header.stamp.fromSec(Measures.lidar_end_time); // ros::Time().fromSec(last_timestamp_lidar);
                    laserCloudMap.header.frame_id = "world";
                    pubLaserCloudMap.publish( laserCloudMap );
                }
                 
                std::vector< bool >               point_selected_surf( feats_down_size, true ); // 记录有那些点成功找到了平面
                std::vector< std::vector< int > > pointSearchInd_surf( feats_down_size );       // 构成平面的点的index
                std::vector< PointVector >        Nearest_Points( feats_down_size );    // 二维数组,存点i的最近点排序后的集合

                int  rematch_num = 0;
                bool rematch_en = 0;

                flg_EKF_converged = 0;
                deltaR = 0.0;
                deltaT = 0.0;
                t2 = omp_get_wtime();
                double maximum_pt_range = 0.0;

               // @note (tim, match_time, laserCloudOri, coeffSel, feats_down_size, feats_down, maximum_pt_range, feats_down_updated, Nearest_Points, rematch_en, point_selected_surf, res_last, pca_time, coeffsel_tmpt, solve_start, H_T_H, Hsub, I_state);
                lio_update(tim, rematch_en, state_propagate, laserCloudOri, coeffSel,feats_down, feats_down_updated, coeffSel_tmpt, feats_down_size, rematch_num, maximum_pt_range, match_time, solve_time, match_start, pca_time, solve_start, Nearest_Points, point_selected_surf, res_last, H_T_H, euler_cur, G, I_STATE);
               //lio_update(tim, rematch_en, state_propagate, laserCloudOri, coeffSel,feats_down, feats_down_updated, coeffSel_tmpt, feats_down_size, rematch_num, maximum_pt_range, match_time, solve_time, match_start, pca_time, solve_start, Nearest_Points, point_selected_surf, res_last, H_T_H, euler_cur, G, I_STATE);
                
                
                t3 = omp_get_wtime();

                PointVector points_history; // 将ikd树中需要移除的点放入points_history中
                ikdtree.acquire_removed_points( points_history );// 从Points_deleted和Multithread_Points_deleted获取点
                memset( cube_updated, 0, sizeof( cube_updated ) );
                //1> : 更新维持的固定大小的map立方体 (参考FAST-LIO2:V.A地图管理)
                for ( int i = 0; i < points_history.size(); i++ )
                {
                    PointType &pointSel = points_history[ i ];

                    int cubeI = int( ( pointSel.x + 0.5 * cube_len ) / cube_len ) + laserCloudCenWidth;
                    int cubeJ = int( ( pointSel.y + 0.5 * cube_len ) / cube_len ) + laserCloudCenHeight;
                    int cubeK = int( ( pointSel.z + 0.5 * cube_len ) / cube_len ) + laserCloudCenDepth;

                    if ( pointSel.x + 0.5 * cube_len < 0 )
                        cubeI--;
                    if ( pointSel.y + 0.5 * cube_len < 0 )
                        cubeJ--;
                    if ( pointSel.z + 0.5 * cube_len < 0 )
                        cubeK--;

                    if ( cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 && cubeK < laserCloudDepth )
                    {
                        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                        featsArray[ cubeInd ]->push_back( pointSel );
                    }
                }     

                //2-1> : 将Kalman更新后的新lidar帧特征点先转世界坐标系
                for ( int i = 0; i < feats_down_size; i++ )
                {
                    /* transform to world frame */
                    pointBodyToWorld( &( feats_down->points[ i ] ), &( feats_down_updated->points[ i ] ) );
                }
                t4 = omp_get_wtime();
                //2-2> : 将特征点加入世界坐标中
                ikdtree.Add_Points( feats_down_updated->points, true ); // 存入ikd树中
                
                kdtree_incremental_time = omp_get_wtime() - t4 + readd_time + readd_box_time + delete_box_time;
                t5 = omp_get_wtime();
            }   // (2-7) ICP迭代+Kalman更新完成 */

            /**
             * @note (2-8) : Publish current frame points in world coordinates:  
             *              发布当前帧的点云数据
             */
            laserCloudFullRes2->clear();
            *laserCloudFullRes2 = dense_map_en ? ( *feats_undistort ) : ( *feats_down );//去畸变or下采样点
            int laserCloudFullResNum = laserCloudFullRes2->points.size();// 发布点数量

            pcl::PointXYZI temp_point;
            laserCloudFullResColor->clear();
            {
                for ( int i = 0; i < laserCloudFullResNum; i++ )
                {   // 将laserCloudFullRes2的点转到世界坐标系下,再存入laserCloudFullResColor
                    RGBpointBodyToWorld( &laserCloudFullRes2->points[ i ], &temp_point );
                    laserCloudFullResColor->push_back( temp_point );
                }
                sensor_msgs::PointCloud2 laserCloudFullRes3;// 将laserCloudFullResColor转为发布形式
                pcl::toROSMsg( *laserCloudFullResColor, laserCloudFullRes3 );
                // laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
                laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time );
                laserCloudFullRes3.header.frame_id = "world"; // world; camera_init
                pubLaserCloudFullRes.publish( laserCloudFullRes3 );
            }

            if ( 1 )
            {
                static std::vector< double > stastic_cost_time;
                Common_tools::Timer          tim;
                // tim.tic();
                // ANCHOR - RGB maps update
                wait_render_thread_finish();
                if ( m_if_record_mvs )
                {
                    std::vector< std::shared_ptr< RGB_pts > > pts_last_hitted;
                    pts_last_hitted.reserve( 1e6 );
                    m_number_of_new_visited_voxel = m_map_rgb_pts.append_points_to_global_map(
                        *laserCloudFullResColor, Measures.lidar_end_time - g_camera_lidar_queue.m_first_imu_time, &pts_last_hitted,
                        m_append_global_map_point_step );
                    m_map_rgb_pts.m_mutex_pts_last_visited->lock();
                    m_map_rgb_pts.m_pts_last_hitted = pts_last_hitted;
                    m_map_rgb_pts.m_mutex_pts_last_visited->unlock();
                }
                else
                {
                    m_number_of_new_visited_voxel = m_map_rgb_pts.append_points_to_global_map(
                        *laserCloudFullResColor, Measures.lidar_end_time - g_camera_lidar_queue.m_first_imu_time, nullptr,
                        m_append_global_map_point_step );
                }
                stastic_cost_time.push_back( tim.toc( " ", 0 ) );
            }

            if(0) // Uncomment this code scope to enable the publish of effective points. 
            {
                /******* Publish effective points *******/
                laserCloudFullResColor->clear();
                pcl::PointXYZI temp_point;
                for ( int i = 0; i < laserCloudSelNum; i++ )
                {
                    RGBpointBodyToWorld( &laserCloudOri->points[ i ], &temp_point );
                    laserCloudFullResColor->push_back( temp_point );
                }
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg( *laserCloudFullResColor, laserCloudFullRes3 );
                // laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
                laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time ); //.fromSec(last_timestamp_lidar);
                laserCloudFullRes3.header.frame_id = "world";
                pubLaserCloudEffect.publish( laserCloudFullRes3 );
            }

            /**
             * @note (2-11)***** Publish Maps:  发布地图******
             */
            sensor_msgs::PointCloud2 laserCloudMap;
            pcl::toROSMsg( *featsFromMap, laserCloudMap );
            laserCloudMap.header.stamp.fromSec( Measures.lidar_end_time ); // ros::Time().fromSec(last_timestamp_lidar);
            laserCloudMap.header.frame_id = "world";
            pubLaserCloudMap.publish( laserCloudMap );

            /**
             * @note (2-12)***** Publish Odometry 发布里程计*****
             */
            geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw( euler_cur( 0 ), euler_cur( 1 ), euler_cur( 2 ) );
            odomAftMapped.header.frame_id = "world";
            odomAftMapped.child_frame_id = "/aft_mapped";
            odomAftMapped.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
            odomAftMapped.pose.pose.orientation.x = geoQuat.x;
            odomAftMapped.pose.pose.orientation.y = geoQuat.y;
            odomAftMapped.pose.pose.orientation.z = geoQuat.z;
            odomAftMapped.pose.pose.orientation.w = geoQuat.w;
            odomAftMapped.pose.pose.position.x = g_lio_state.pos_end( 0 );
            odomAftMapped.pose.pose.position.y = g_lio_state.pos_end( 1 );
            odomAftMapped.pose.pose.position.z = g_lio_state.pos_end( 2 );
            pubOdomAftMapped.publish( odomAftMapped );

            static tf::TransformBroadcaster br;
            tf::Transform                   transform;
            tf::Quaternion                  q;
            transform.setOrigin(
                tf::Vector3( odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z ) );
            q.setW( odomAftMapped.pose.pose.orientation.w );
            q.setX( odomAftMapped.pose.pose.orientation.x );
            q.setY( odomAftMapped.pose.pose.orientation.y );
            q.setZ( odomAftMapped.pose.pose.orientation.z );
            transform.setRotation( q );
            br.sendTransform( tf::StampedTransform( transform, ros::Time().fromSec( Measures.lidar_end_time ), "world", "/aft_mapped" ) );

            msg_body_pose.header.stamp = ros::Time::now();
            msg_body_pose.header.frame_id = "/camera_odom_frame";
            msg_body_pose.pose.position.x = g_lio_state.pos_end( 0 );
            msg_body_pose.pose.position.y = g_lio_state.pos_end( 1 );
            msg_body_pose.pose.position.z = g_lio_state.pos_end( 2 );
            msg_body_pose.pose.orientation.x = geoQuat.x;
            msg_body_pose.pose.orientation.y = geoQuat.y;
            msg_body_pose.pose.orientation.z = geoQuat.z;
            msg_body_pose.pose.orientation.w = geoQuat.w;
            msg_body_pose.header.frame_id = "world";

            if ( frame_num > 10 )
            {
                path.poses.push_back( msg_body_pose );
            }
            pubPath.publish( path );
            

            frame_num++;
            aver_time_consu = aver_time_consu * ( frame_num - 1 ) / frame_num + ( t5 - t0 ) / frame_num;
            // aver_time_consu = aver_time_consu * 0.8 + (t5 - t0) * 0.2;
            T1[ time_log_counter ] = Measures.lidar_beg_time;
            s_plot[ time_log_counter ] = aver_time_consu;
            s_plot2[ time_log_counter ] = kdtree_incremental_time;
            s_plot3[ time_log_counter ] = kdtree_search_time;
            s_plot4[ time_log_counter ] = fov_check_time;
            s_plot5[ time_log_counter ] = t5 - t0;
            s_plot6[ time_log_counter ] = readd_box_time;
            time_log_counter++;
            fprintf( m_lio_costtime_fp, "%.5f %.5f\r\n", g_lio_state.last_update_time - g_camera_lidar_queue.m_first_imu_time, t5 - t0 );
            fflush( m_lio_costtime_fp );
        }
        status = ros::ok();
        rate.sleep();
    }
    return 0;
}
