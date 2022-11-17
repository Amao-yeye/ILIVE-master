#include "ilive.hpp"


void ILIVE::imu_cbk( const sensor_msgs::Imu::ConstPtr &msg_in )
{
    sensor_msgs::Imu::Ptr msg( new sensor_msgs::Imu( *msg_in ) );
    double                timestamp = msg->header.stamp.toSec(); // 获取IMU的当前时间
    g_camera_lidar_queue.imu_in( timestamp );   // 设置IMU的当前时刻时间
    mtx_buffer.lock();  // imu上锁
    if ( timestamp < last_timestamp_imu )   // last_timestamp_imu初始为-1
    {
        ROS_ERROR( "imu loop back, clear buffer" );
        imu_buffer_lio.clear();
        imu_buffer_vio.clear();
        flg_reset = true;
    }
    last_timestamp_imu = timestamp;

    if ( g_camera_lidar_queue.m_if_acc_mul_G )
    {
        msg->linear_acceleration.x *= G_m_s2;
        msg->linear_acceleration.y *= G_m_s2;
        msg->linear_acceleration.z *= G_m_s2;
    }

    //向lio和vio中输入imu数据
    imu_buffer_lio.push_back( msg );
    imu_buffer_vio.push_back( msg );

    mtx_buffer.unlock();    // imu解锁
    sig_buffer.notify_all();
}

void printf_field_name( sensor_msgs::PointCloud2::ConstPtr &msg )
{
    cout << "Input pointcloud field names: [" << msg->fields.size() << "]: ";
    for ( size_t i = 0; i < msg->fields.size(); i++ )
    {
        cout << msg->fields[ i ].name << ", ";
    }
    cout << endl;
}



void ILIVE::load_lio_parameters()
{

    std::vector<double> t_lid_in_imu, R_lid_in_imu;
    m_ros_node_handle.getParam("ilive_lio/ext_t_lid_in_imu", t_lid_in_imu);
    m_ros_node_handle.getParam("ilive_lio/ext_R_lid_in_imu", R_lid_in_imu);



    if ( (R_lid_in_imu.size() != 9) || (t_lid_in_imu.size() != 3))
    {

        cout << ANSI_COLOR_RED_BOLD << "Load LIO parameter fail!!!, please check!!!" << endl;
        std::this_thread::sleep_for(std::chrono::seconds(3000000));
    }

    ext_R_lid_in_imu = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_lid_in_imu.data());
    ext_t_lid_in_imu = Eigen::Map<Eigen::Matrix<double, 3, 1>>(t_lid_in_imu.data());
}



bool ILIVE::get_pointcloud_data_from_ros_message( sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud< pcl::PointXYZINormal > &pcl_pc )
{

    // printf("Frame [%d] %.3f ", g_LiDAR_frame_index,  msg->header.stamp.toSec() - g_camera_lidar_queue.m_first_imu_time);
    pcl::PointCloud< pcl::PointXYZI > res_pc;
    scope_color( ANSI_COLOR_YELLOW_BOLD );
    // printf_field_name(msg);
    if ( msg->fields.size() < 3 )
    {
        cout << "Get pointcloud data from ros messages fail!!!" << endl;
        scope_color( ANSI_COLOR_RED_BOLD );
        printf_field_name( msg );
        return false;
    }
    else
    {
        if ( ( msg->fields.size() == 8 ) && ( msg->fields[ 3 ].name == "intensity" ) &&
             ( msg->fields[ 4 ].name == "normal_x" ) ) // Input message type is pcl::PointXYZINormal
        {
            pcl::fromROSMsg( *msg, pcl_pc );
            return true;
        }
        else if ( ( msg->fields.size() == 4 ) && ( msg->fields[ 3 ].name == "rgb" ) )
        {
            double maximum_range = 5;
            get_ros_parameter< double >( m_ros_node_handle, "iros_range", maximum_range, 5 );
            pcl::PointCloud< pcl::PointXYZRGB > pcl_rgb_pc;
            pcl::fromROSMsg( *msg, pcl_rgb_pc );
            double lidar_point_time = msg->header.stamp.toSec();
            int    pt_count = 0;
            pcl_pc.resize( pcl_rgb_pc.points.size() );
            for ( int i = 0; i < pcl_rgb_pc.size(); i++ )
            {
                pcl::PointXYZINormal temp_pt;
                temp_pt.x = pcl_rgb_pc.points[ i ].x;
                temp_pt.y = pcl_rgb_pc.points[ i ].y;
                temp_pt.z = pcl_rgb_pc.points[ i ].z;
                double frame_dis = sqrt( temp_pt.x * temp_pt.x + temp_pt.y * temp_pt.y + temp_pt.z * temp_pt.z );
                if ( frame_dis > maximum_range )
                {
                    continue;
                }
                temp_pt.intensity = ( pcl_rgb_pc.points[ i ].r + pcl_rgb_pc.points[ i ].g + pcl_rgb_pc.points[ i ].b ) / 3.0;
                temp_pt.curvature = 0;
                pcl_pc.points[ pt_count ] = temp_pt;
                pt_count++;
            }
            pcl_pc.points.resize( pt_count );
            return true;
        }
        else // TODO, can add by yourself
        {
            cout << "Get pointcloud data from ros messages fail!!! ";
            scope_color( ANSI_COLOR_RED_BOLD );
            printf_field_name( msg );
            return false;
        }
    }
}


bool ILIVE::sync_packages( MeasureGroup &meas )
{
    //(1)判断lidar和imu的数据buffer是否为空
    if ( lidar_buffer.empty() || imu_buffer_lio.empty() )
    {
        return false;
    }

    /*** push lidar frame 取出lidar数据***/
    if ( !lidar_pushed )    // lidar_pushed初始false
    {
        meas.lidar.reset( new PointCloudXYZINormal() );
        // 从lidar_buffer中获取点云数据(出队列: 取队列队首数据)
        if ( get_pointcloud_data_from_ros_message( lidar_buffer.front(), *( meas.lidar ) ) == false )
        {
            return false;
        }
        // pcl::fromROSMsg(*(lidar_buffer.front()), *(meas.lidar));
        meas.lidar_beg_time = lidar_buffer.front()->header.stamp.toSec();   // 当前lidar帧的开始时间
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double( 1000 );
        meas.lidar_end_time = lidar_end_time;   // 当前lidar帧的结束时间
        lidar_pushed = true;
    }

    if ( last_timestamp_imu < lidar_end_time )
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer 取出imu数据***/
    double imu_time = imu_buffer_lio.front()->header.stamp.toSec();
    meas.imu.clear();
    while ( ( !imu_buffer_lio.empty() ) && ( imu_time < lidar_end_time ) )
    {
        imu_time = imu_buffer_lio.front()->header.stamp.toSec();
        if ( imu_time > lidar_end_time + 0.02 )
            break;
        meas.imu.push_back( imu_buffer_lio.front() );
        imu_buffer_lio.pop_front();
    }

    lidar_buffer.pop_front();
    lidar_pushed = false;
    // if (meas.imu.empty()) return false;
 
    return true;
}

// project lidar frame to world - 将参数1-pi转换至世界坐标系下,并赋值给参数2-po
void ILIVE::pointBodyToWorld( PointType const *const pi, PointType *const po )
{
    Eigen::Vector3d p_body( pi->x, pi->y, pi->z );
    Eigen::Vector3d p_global( g_lio_state.rot_end * ( p_body + ext_t_lid_in_imu ) + g_lio_state.pos_end );

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;
}

void ILIVE::RGBpointBodyToWorld( PointType const *const pi, pcl::PointXYZI *const po )
{
    Eigen::Vector3d p_body( pi->x, pi->y, pi->z );
    Eigen::Vector3d p_global( g_lio_state.rot_end * ( p_body + ext_t_lid_in_imu ) + g_lio_state.pos_end );

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - std::floor( intensity );

    int reflection_map = intensity * 10000;
}

/**
 * @note 获取立方体中第k个面的第j行第i列的下标
 */ 
int ILIVE::get_cube_index( const int &i, const int &j, const int &k )
{
    return ( i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k );
}

bool ILIVE::center_in_FOV( Eigen::Vector3f cube_p )
{
    Eigen::Vector3f dis_vec = g_lio_state.pos_end.cast< float >() - cube_p;
    float           squaredSide1 = dis_vec.transpose() * dis_vec;

    if ( squaredSide1 < 0.4 * cube_len * cube_len )
        return true;

    dis_vec = XAxisPoint_world.cast< float >() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;

    float ang_cos =
        fabs( squaredSide1 <= 3 ) ? 1.0 : ( LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2 ) / ( 2 * LIDAR_SP_LEN * sqrt( squaredSide1 ) );

    return ( ( ang_cos > HALF_FOV_COS ) ? true : false );
}

bool ILIVE::if_corner_in_FOV( Eigen::Vector3f cube_p )
{
    Eigen::Vector3f dis_vec = g_lio_state.pos_end.cast< float >() - cube_p;
    float           squaredSide1 = dis_vec.transpose() * dis_vec;
    dis_vec = XAxisPoint_world.cast< float >() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;
    float ang_cos =
        fabs( squaredSide1 <= 3 ) ? 1.0 : ( LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2 ) / ( 2 * LIDAR_SP_LEN * sqrt( squaredSide1 ) );
    return ( ( ang_cos > HALF_FOV_COS ) ? true : false );
}


void ILIVE::lasermap_fov_segment()
{
    laserCloudValidNum = 0;
    pointBodyToWorld( XAxisPoint_body, XAxisPoint_world );// 将lidar坐标系变换至世界坐标系

    int centerCubeI = int( ( g_lio_state.pos_end( 0 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenWidth;
    int centerCubeJ = int( ( g_lio_state.pos_end( 1 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenHeight;
    int centerCubeK = int( ( g_lio_state.pos_end( 2 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenDepth;
    if ( g_lio_state.pos_end( 0 ) + 0.5 * cube_len < 0 )
        centerCubeI--;
    if ( g_lio_state.pos_end( 1 ) + 0.5 * cube_len < 0 )
        centerCubeJ--;
    if ( g_lio_state.pos_end( 2 ) + 0.5 * cube_len < 0 )
        centerCubeK--;
    bool last_inFOV_flag = 0;
    int  cube_index = 0;
    cub_needrm.clear();
    cub_needad.clear();
    T2[ time_log_counter ] = Measures.lidar_beg_time;
    double t_begin = omp_get_wtime();

    // 计算立方体内每个体素的点信息
    while ( centerCubeI < FOV_RANGE + 1 )
    {
        for ( int j = 0; j < laserCloudHeight; j++ ) // laserCloudHeight=48
        {
            for ( int k = 0; k < laserCloudDepth; k++ ) // laserCloudDepth=48
            {   
                // 下面用到的get_cube_index(i, j, k) : 获取立方体中第k个面的第j行第i列的下标

                int i = laserCloudWidth - 1;    // laserCloudWidth=48

                // 单独获取(47,0,0)处的cube信息 (featsArray[48*48*48])
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i - 1, j, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i - 1, j, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeI++;
        laserCloudCenWidth++;
    }
 
    while ( centerCubeI >= laserCloudWidth - ( FOV_RANGE + 1 ) )
    {
        for ( int j = 0; j < laserCloudHeight; j++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int i = 0;

                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )   // 因为上面定义i=0,因此不进入循环
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i + 1, j, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i + 1, j, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeI--;
        laserCloudCenWidth--;
    }

    while ( centerCubeJ < ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int j = laserCloudHeight - 1; // laserCloudHeight=48

                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j - 1, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j - 1, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
    }

    while ( centerCubeJ >= laserCloudHeight - ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int                       j = 0;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j + 1, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j + 1, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
    }

    while ( centerCubeK < ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int j = 0; j < laserCloudHeight; j++ )
            {
                int                       k = laserCloudDepth - 1;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j, k - 1 ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j, k - 1 ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK++;
        laserCloudCenDepth++;
    }

    while ( centerCubeK >= laserCloudDepth - ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int j = 0; j < laserCloudHeight; j++ )
            {
                int                       k = 0;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j, k + 1 ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j, k + 1 ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeK--;
        laserCloudCenDepth--;
    }

    cube_points_add->clear();
    featsFromMap->clear();
    memset( now_inFOV, 0, sizeof( now_inFOV ) );
    copy_time = omp_get_wtime() - t_begin;
    double fov_check_begin = omp_get_wtime();

    fov_check_time = omp_get_wtime() - fov_check_begin;

    double readd_begin = omp_get_wtime();
#ifdef USE_ikdtree
    if ( cub_needrm.size() > 0 )
        ikdtree.Delete_Point_Boxes( cub_needrm );
    delete_box_time = omp_get_wtime() - readd_begin;
    // s_plot4.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if ( cub_needad.size() > 0 )
        ikdtree.Add_Point_Boxes( cub_needad );
    readd_box_time = omp_get_wtime() - readd_begin - delete_box_time;
    // s_plot5.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if ( cube_points_add->points.size() > 0 )
        ikdtree.Add_Points( cube_points_add->points, true );
#endif
    readd_time = omp_get_wtime() - readd_begin - delete_box_time - readd_box_time;
    // s_plot6.push_back(omp_get_wtime() - t_begin);
}
 
void ILIVE::feat_points_cbk( const sensor_msgs::PointCloud2::ConstPtr &msg_in )
{
    sensor_msgs::PointCloud2::Ptr msg( new sensor_msgs::PointCloud2( *msg_in ) );
    msg->header.stamp = ros::Time( msg_in->header.stamp.toSec() - m_lidar_imu_time_delay );
    if ( g_camera_lidar_queue.lidar_in( msg_in->header.stamp.toSec() + 0.1 ) == 0 )
    {
        return;
    }
    mtx_buffer.lock();  // imu上锁
    // std::cout<<"got feature"<<std::endl;
    if ( msg->header.stamp.toSec() < last_timestamp_lidar )
    {   // last_timestamp_lidar初始=-1
        ROS_ERROR( "lidar loop back, clear buffer" );
        lidar_buffer.clear();
    }

    lidar_buffer.push_back( msg );
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();        // imu解锁
    sig_buffer.notify_all();    // 唤醒LIO处理线程,进行Lidar数据处理
}

void ILIVE::wait_render_thread_finish()
{
    if ( m_render_thread != nullptr )
    {
        m_render_thread->get(); // wait render thread to finish.
    }
}
