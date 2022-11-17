#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include "../tools/tools_logger.hpp"

using namespace std;

#define IS_VALID( a ) ( ( abs( a ) > 1e8 ) ? true : false )

typedef pcl::PointXYZINormal PointType;

ros::Publisher pub_full, pub_surf, pub_corn;

enum LID_TYPE
{
    HORIZON,
    VELO16
};

enum Feature
{
    Nor,
    Poss_Plane,
    Real_Plane,
    Edge_Jump,
    Edge_Plane,
    Wire,
    ZeroPoint
};
enum Surround
{
    Prev,
    Next
};
enum E_jump
{
    Nr_nor,
    Nr_zero,
    Nr_180,
    Nr_inf,
    Nr_blind
};

struct orgtype
{
    double  range;
    double  dista;
    double  angle[ 2 ];
    double  intersect;
    E_jump  edj[ 2 ];   //枚举类型
    Feature ftype;      //枚举类型
    orgtype()   // 构造函数
    {
        range = 0;// Lidar点的深度信息
        edj[ Prev ] = Nr_nor;   // 0
        edj[ Next ] = Nr_nor;   // 0
        ftype = Nor;            // 0
        intersect = 2;
    }
};

const double rad2deg = 180 * M_1_PI;

int    lidar_type;
double blind, inf_bound;
int    N_SCANS;
int    group_size;
double disA, disB;
double limit_maxmid, limit_midmin, limit_maxmin;
double p2l_ratio;
double jump_up_limit, jump_down_limit;
double cos160;
double edgea, edgeb;
double smallp_intersect, smallp_ratio;
int    point_filter_num;
int    g_if_using_raw_point = 1;
int    g_LiDAR_sampling_point_step = 3;
void   velo16_handler( const sensor_msgs::PointCloud2::ConstPtr &msg );
void   give_feature( pcl::PointCloud< PointType > &pl, vector< orgtype > &types, pcl::PointCloud< PointType > &pl_corn,
                     pcl::PointCloud< PointType > &pl_surf );
void   pub_func( pcl::PointCloud< PointType > &pl, ros::Publisher pub, const ros::Time &ct );
int    plane_judge( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct );
bool   small_plane( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct );
bool   edge_jump_judge( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i, Surround nor_dir );

int main( int argc, char **argv )
{
    /**
     * (1)初始化名为"feature_extract"的ros节点
     */ 
    ros::init( argc, argv, "feature_extract" );
    ros::NodeHandle n;

    /**
     * (2)从参数服务器上获取参数,如果获取失败则=默认值
     * @note NodeHandle::param(const std::string& param_name, T& param_val, const T& default_val)
     */ 
    n.param< int >( "Lidar_front_end/lidar_type", lidar_type, 0 );
    n.param< double >( "Lidar_front_end/blind", blind, 0.1 );
    n.param< double >( "Lidar_front_end/inf_bound", inf_bound, 4 );
    n.param< int >( "Lidar_front_end/N_SCANS", N_SCANS, 1 );
    n.param< int >( "Lidar_front_end/group_size", group_size, 8 );
    n.param< double >( "Lidar_front_end/disA", disA, 0.01 );
    n.param< double >( "Lidar_front_end/disB", disB, 0.1 );
    n.param< double >( "Lidar_front_end/p2l_ratio", p2l_ratio, 225 );
    n.param< double >( "Lidar_front_end/limit_maxmid", limit_maxmid, 6.25 );
    n.param< double >( "Lidar_front_end/limit_midmin", limit_midmin, 6.25 );
    n.param< double >( "Lidar_front_end/limit_maxmin", limit_maxmin, 3.24 );
    n.param< double >( "Lidar_front_end/jump_up_limit", jump_up_limit, 170.0 );
    n.param< double >( "Lidar_front_end/jump_down_limit", jump_down_limit, 8.0 );
    n.param< double >( "Lidar_front_end/cos160", cos160, 160.0 );
    n.param< double >( "Lidar_front_end/edgea", edgea, 2 );
    n.param< double >( "Lidar_front_end/edgeb", edgeb, 0.1 );
    n.param< double >( "Lidar_front_end/smallp_intersect", smallp_intersect, 172.5 );
    n.param< double >( "Lidar_front_end/smallp_ratio", smallp_ratio, 1.2 );
    n.param< int >( "Lidar_front_end/point_filter_num", point_filter_num, 1 );
    n.param< int >( "Lidar_front_end/point_step", g_LiDAR_sampling_point_step, 3 );
    n.param< int >( "Lidar_front_end/using_raw_point", g_if_using_raw_point, 1 );

    // 设置需要用到的全局变量 - cos(角度->弧度)
    jump_up_limit = cos( jump_up_limit / 180 * M_PI );      // cos(170度)
    jump_down_limit = cos( jump_down_limit / 180 * M_PI );  // cos(8度)
    cos160 = cos( cos160 / 180 * M_PI );                    // cos(160度)
    smallp_intersect = cos( smallp_intersect / 180 * M_PI );// cos(172.5度)

    ros::Subscriber sub_points;

    /**
     * (3)根据lidar_type的值进行lidar数据的订阅设置
     *      绑定订阅后的回调函数
     */ 
    switch ( lidar_type )
    {
    case HORIZON:   // enum HORIZON = 0 : r3live_bag_ouster.launch中设置
        printf( "HORIZON\n" );
        //sub_points = n.subscribe( "/livox/lidar", 1000, horizon_handler, ros::TransportHints().tcpNoDelay() );
        break;

    case VELO16:
        printf( "VELO16\n" );   // enum VELO16 = 1 : TODO
        sub_points = n.subscribe( "/velodyne_points", 1000, velo16_handler, ros::TransportHints().tcpNoDelay() );
        break;

    default:
        printf( "Lidar type is wrong.\n" );
        exit( 0 );
        break;
    }

    /**
     * (4)设置需要发布的消息格式
     *      回调函数接收到原始点云后,将其转换为指定格式,并提取角点特征和面特征
     *      然后通过pub_full,pub_surf,pub_corn将消息发布出去
     */ 
    pub_full = n.advertise< sensor_msgs::PointCloud2 >( "/laser_cloud", 100 );  //发布转换后的消息
    pub_surf = n.advertise< sensor_msgs::PointCloud2 >( "/laser_cloud_flat", 100 ); //发布面特征消息
    pub_corn = n.advertise< sensor_msgs::PointCloud2 >( "/laser_cloud_sharp", 100 );//发布角点特征消息

    ros::spin();
    return 0;
}

/**
 * @note LiDAR_front_end的回调函数
 *      接收/livox/lidar原始数据,回调函数中转换数据(/laser_cloud),
 *      并提取角/面特征后发布出去(laser_cloud_sharp/laser_cloud_flat)
 */ 
double vx, vy, vz;



namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      float time;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (std::uint16_t, ring, ring)
)

void velo16_handler( const sensor_msgs::PointCloud2::ConstPtr &msg )
{
    // typedef pcl::PointXYZINormal PointType;
    // float x, y, z, intensity, normal[3]: normal_x, normal_y, normal_z, curvature;

    pcl::PointCloud< PointType > pl_processed;  // 存放处理后Lidar点的一维数组

    pcl::PointCloud<velodyne_ros::Point> pl_orig;  // 存放Lidar点的一维数组
    pcl::fromROSMsg(*msg, pl_orig); // 通过PointCloud2格式的msg初始化PointXYZINormal的pl_orig
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    // pl_surf.reserve(plsize);

    double time_stamp = msg->header.stamp.toSec(); // 
    pl_processed.clear();
    pl_processed.reserve( pl_orig.points.size() ); // 预留size


    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * 10;        		// SCAN_RATE = 10; scan angular velocity  
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      	// yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   	// yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  	// last offset time
    /*****************************************************************/

    bool given_offset_time = false;			//add definition for 'given_offset_time'
    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;
    }
    else
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * 1.f;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time)
        {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer])  // 
            {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
                yaw_fp[layer]=yaw_angle;
                is_first[layer]=false;
                added_pt.curvature = 0.0;
                yaw_last[layer]=yaw_angle;
                time_last[layer]=added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer])
            {
                added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
            }
            else
            {
                added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer]=added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
            if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
            {
                pl_processed.points.push_back(added_pt);
            }
        }
    }
    pub_func( pl_processed, pub_full, msg->header.stamp );
    pub_func( pl_processed, pub_surf, msg->header.stamp );
    pub_func( pl_processed, pub_corn, msg->header.stamp );
}

void give_feature( pcl::PointCloud< PointType > &pl, vector< orgtype > &types, 
                   pcl::PointCloud< PointType > &pl_corn,
                   pcl::PointCloud< PointType > &pl_surf )
{
    /**
     * @note (1)预先判断点数量和点的距离,并定义相关变量
     */
    uint plsize = pl.size();
    uint plsize2;
    if ( plsize == 0 )  // 判断点数量
    {
        printf( "something wrong\n" );
        return;
    }
    uint head = 0;
    while ( types[ head ].range < blind )   // blind = 0.1 : default
    {           // types[i].range = pl[i].x
        head++; // 因此猜测这里判断lidar帧中点的距离
    }           // 保证后续处理的起点在一定距离外,防止干扰点

    // Surf : group_size = 8 : 点数大于8,则后续处理范围为 head -> 点数-8,
    // 否则为0, 因该是只计算head到倒数第8点之间的中间点.
    plsize2 = ( plsize > group_size ) ? ( plsize - group_size ) : 0;

    Eigen::Vector3d curr_direct( Eigen::Vector3d::Zero() );
    Eigen::Vector3d last_direct( Eigen::Vector3d::Zero() );

    uint i_nex = 0, i2;
    uint last_i = 0;
    uint last_i_nex = 0;
    int  last_state = 0;
    int  plane_type;

    PointType ap;// 充当中间交换量的临时容器
    /**
     * @note (2)
     *  对head到倒数第8点之间的中间点进行计算
     */
    for ( uint i = head; i < plsize2; i += g_LiDAR_sampling_point_step )    // g_LiDAR_sampling_point_step = 3
    {
        //
        if ( types[ i ].range > blind )
        {
            ap.x = pl[ i ].x;
            ap.y = pl[ i ].y;
            ap.z = pl[ i ].z;
            ap.curvature = pl[ i ].curvature;
            pl_surf.push_back( ap );
        }
        if ( g_if_using_raw_point )
        {
            continue;
        }
        // i_nex = i;
        i2 = i;
        // std::cout<<" i: "<<i<<" i_nex "<<i_nex<<"group_size: "<<group_size<<" plsize "<<plsize<<" plsize2
        // "<<plsize2<<std::endl;
        plane_type = plane_judge( pl, types, i, i_nex, curr_direct );

        if ( plane_type == 1 )
        {
            for ( uint j = i; j <= i_nex; j++ )
            {
                if ( j != i && j != i_nex )
                {
                    types[ j ].ftype = Real_Plane;
                }
                else
                {
                    types[ j ].ftype = Poss_Plane;
                }
            }

            // if(last_state==1 && fabs(last_direct.sum())>0.5)
            if ( last_state == 1 && last_direct.norm() > 0.1 )
            {
                double mod = last_direct.transpose() * curr_direct;
                if ( mod > -0.707 && mod < 0.707 )
                {
                    types[ i ].ftype = Edge_Plane;
                }
                else
                {
                    types[ i ].ftype = Real_Plane;
                }
            }

            i = i_nex - 1;
            last_state = 1;
        }
        else if ( plane_type == 2 )
        {
            i = i_nex;
            last_state = 0;
        }
        else if ( plane_type == 0 )
        {
            if ( last_state == 1 )
            {
                uint i_nex_tem;
                uint j;
                for ( j = last_i + 1; j <= last_i_nex; j++ )
                {
                    uint            i_nex_tem2 = i_nex_tem;
                    Eigen::Vector3d curr_direct2;

                    uint ttem = plane_judge( pl, types, j, i_nex_tem, curr_direct2 );

                    if ( ttem != 1 )
                    {
                        i_nex_tem = i_nex_tem2;
                        break;
                    }
                    curr_direct = curr_direct2;
                }

                if ( j == last_i + 1 )
                {
                    last_state = 0;
                }
                else
                {
                    for ( uint k = last_i_nex; k <= i_nex_tem; k++ )
                    {
                        if ( k != i_nex_tem )
                        {
                            types[ k ].ftype = Real_Plane;
                        }
                        else
                        {
                            types[ k ].ftype = Poss_Plane;
                        }
                    }
                    i = i_nex_tem - 1;
                    i_nex = i_nex_tem;
                    i2 = j - 1;
                    last_state = 1;
                }
            }
        }

        last_i = i2;
        last_i_nex = i_nex;
        last_direct = curr_direct;
    }
    if ( g_if_using_raw_point )
    {
        return;
    }
    plsize2 = plsize > 3 ? plsize - 3 : 0;
    for ( uint i = head + 3; i < plsize2; i++ )
    {
        if ( types[ i ].range < blind || types[ i ].ftype >= Real_Plane )
        {
            continue;
        }

        if ( types[ i - 1 ].dista < 1e-16 || types[ i ].dista < 1e-16 )
        {
            continue;
        }

        Eigen::Vector3d vec_a( pl[ i ].x, pl[ i ].y, pl[ i ].z );
        Eigen::Vector3d vecs[ 2 ];

        for ( int j = 0; j < 2; j++ )
        {
            int m = -1;
            if ( j == 1 )
            {
                m = 1;
            }

            if ( types[ i + m ].range < blind )
            {
                if ( types[ i ].range > inf_bound )
                {
                    types[ i ].edj[ j ] = Nr_inf;
                }
                else
                {
                    types[ i ].edj[ j ] = Nr_blind;
                }
                continue;
            }

            vecs[ j ] = Eigen::Vector3d( pl[ i + m ].x, pl[ i + m ].y, pl[ i + m ].z );
            vecs[ j ] = vecs[ j ] - vec_a;

            types[ i ].angle[ j ] = vec_a.dot( vecs[ j ] ) / vec_a.norm() / vecs[ j ].norm();
            if ( types[ i ].angle[ j ] < jump_up_limit )
            {
                types[ i ].edj[ j ] = Nr_180;
            }
            else if ( types[ i ].angle[ j ] > jump_down_limit )
            {
                types[ i ].edj[ j ] = Nr_zero;
            }
        }

        types[ i ].intersect = vecs[ Prev ].dot( vecs[ Next ] ) / vecs[ Prev ].norm() / vecs[ Next ].norm();
        if ( types[ i ].edj[ Prev ] == Nr_nor && types[ i ].edj[ Next ] == Nr_zero && types[ i ].dista > 0.0225 &&
             types[ i ].dista > 4 * types[ i - 1 ].dista )
        {
            if ( types[ i ].intersect > cos160 )
            {
                if ( edge_jump_judge( pl, types, i, Prev ) )
                {
                    types[ i ].ftype = Edge_Jump;
                }
            }
        }
        else if ( types[ i ].edj[ Prev ] == Nr_zero && types[ i ].edj[ Next ] == Nr_nor && types[ i - 1 ].dista > 0.0225 &&
                  types[ i - 1 ].dista > 4 * types[ i ].dista )
        {
            if ( types[ i ].intersect > cos160 )
            {
                if ( edge_jump_judge( pl, types, i, Next ) )
                {
                    types[ i ].ftype = Edge_Jump;
                }
            }
        }
        else if ( types[ i ].edj[ Prev ] == Nr_nor && types[ i ].edj[ Next ] == Nr_inf )
        {
            if ( edge_jump_judge( pl, types, i, Prev ) )
            {
                types[ i ].ftype = Edge_Jump;
            }
        }
        else if ( types[ i ].edj[ Prev ] == Nr_inf && types[ i ].edj[ Next ] == Nr_nor )
        {
            if ( edge_jump_judge( pl, types, i, Next ) )
            {
                types[ i ].ftype = Edge_Jump;
            }
        }
        else if ( types[ i ].edj[ Prev ] > Nr_nor && types[ i ].edj[ Next ] > Nr_nor )
        {
            if ( types[ i ].ftype == Nor )
            {
                types[ i ].ftype = Wire;
            }
        }
    }

    plsize2 = plsize - 1;
    double ratio;
    for ( uint i = head + 1; i < plsize2; i++ )
    {
        if ( types[ i ].range < blind || types[ i - 1 ].range < blind || types[ i + 1 ].range < blind )
        {
            continue;
        }

        if ( types[ i - 1 ].dista < 1e-8 || types[ i ].dista < 1e-8 )
        {
            continue;
        }

        if ( types[ i ].ftype == Nor )
        {
            if ( types[ i - 1 ].dista > types[ i ].dista )
            {
                ratio = types[ i - 1 ].dista / types[ i ].dista;
            }
            else
            {
                ratio = types[ i ].dista / types[ i - 1 ].dista;
            }

            if ( types[ i ].intersect < smallp_intersect && ratio < smallp_ratio )
            {
                if ( types[ i - 1 ].ftype == Nor )
                {
                    types[ i - 1 ].ftype = Real_Plane;
                }
                if ( types[ i + 1 ].ftype == Nor )
                {
                    types[ i + 1 ].ftype = Real_Plane;
                }
                types[ i ].ftype = Real_Plane;
            }
        }
    }

    int last_surface = -1;
    for ( uint j = head; j < plsize; j++ )
    {
        if ( types[ j ].ftype == Poss_Plane || types[ j ].ftype == Real_Plane )
        {
            if ( last_surface == -1 )
            {
                last_surface = j;
            }

            if ( j == uint( last_surface + point_filter_num - 1 ) )
            {
                PointType ap;
                ap.x = pl[ j ].x;
                ap.y = pl[ j ].y;
                ap.z = pl[ j ].z;
                ap.curvature = pl[ j ].curvature;
                pl_surf.push_back( ap );

                last_surface = -1;
            }
        }
        else
        {
            if ( types[ j ].ftype == Edge_Jump || types[ j ].ftype == Edge_Plane )
            {
                pl_corn.push_back( pl[ j ] );
            }
            if ( last_surface != -1 )
            {
                PointType ap;
                for ( uint k = last_surface; k < j; k++ )
                {
                    ap.x += pl[ k ].x;
                    ap.y += pl[ k ].y;
                    ap.z += pl[ k ].z;
                    ap.curvature += pl[ k ].curvature;
                }
                ap.x /= ( j - last_surface );
                ap.y /= ( j - last_surface );
                ap.z /= ( j - last_surface );
                ap.curvature /= ( j - last_surface );
                pl_surf.push_back( ap );
            }
            last_surface = -1;
        }
    }
}

void pub_func( pcl::PointCloud< PointType > &pl, ros::Publisher pub, const ros::Time &ct )
{
    pl.height = 1;
    pl.width = pl.size();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg( pl, output );
    output.header.frame_id = "velodyne";
    output.header.stamp = ct;
    pub.publish( output );
}

int plane_judge( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct )
{
    double group_dis = disA * types[ i_cur ].range + disB;
    group_dis = group_dis * group_dis;
    // i_nex = i_cur;

    double           two_dis;
    vector< double > disarr;
    disarr.reserve( 20 );

    for ( i_nex = i_cur; i_nex < i_cur + group_size; i_nex++ )
    {
        if ( types[ i_nex ].range < blind )
        {
            curr_direct.setZero();
            return 2;
        }
        disarr.push_back( types[ i_nex ].dista );
    }

    for ( ;; )
    {
        if ( ( i_cur >= pl.size() ) || ( i_nex >= pl.size() ) )
            break;

        if ( types[ i_nex ].range < blind )
        {
            curr_direct.setZero();
            return 2;
        }
        vx = pl[ i_nex ].x - pl[ i_cur ].x;
        vy = pl[ i_nex ].y - pl[ i_cur ].y;
        vz = pl[ i_nex ].z - pl[ i_cur ].z;
        two_dis = vx * vx + vy * vy + vz * vz;
        if ( two_dis >= group_dis )
        {
            break;
        }
        disarr.push_back( types[ i_nex ].dista );
        i_nex++;
    }

    double leng_wid = 0;
    double v1[ 3 ], v2[ 3 ];
    for ( uint j = i_cur + 1; j < i_nex; j++ )
    {
        if ( ( j >= pl.size() ) || ( i_cur >= pl.size() ) )
            break;
        v1[ 0 ] = pl[ j ].x - pl[ i_cur ].x;
        v1[ 1 ] = pl[ j ].y - pl[ i_cur ].y;
        v1[ 2 ] = pl[ j ].z - pl[ i_cur ].z;

        v2[ 0 ] = v1[ 1 ] * vz - vy * v1[ 2 ];
        v2[ 1 ] = v1[ 2 ] * vx - v1[ 0 ] * vz;
        v2[ 2 ] = v1[ 0 ] * vy - vx * v1[ 1 ];

        double lw = v2[ 0 ] * v2[ 0 ] + v2[ 1 ] * v2[ 1 ] + v2[ 2 ] * v2[ 2 ];
        if ( lw > leng_wid )
        {
            leng_wid = lw;
        }
    }

    if ( ( two_dis * two_dis / leng_wid ) < p2l_ratio )
    {
        curr_direct.setZero();
        return 0;
    }

    uint disarrsize = disarr.size();
    for ( uint j = 0; j < disarrsize - 1; j++ )
    {
        for ( uint k = j + 1; k < disarrsize; k++ )
        {
            if ( disarr[ j ] < disarr[ k ] )
            {
                leng_wid = disarr[ j ];
                disarr[ j ] = disarr[ k ];
                disarr[ k ] = leng_wid;
            }
        }
    }

    if ( disarr[ disarr.size() - 2 ] < 1e-16 )
    {
        curr_direct.setZero();
        return 0;
    }

    if ( lidar_type == HORIZON )
    {
        double dismax_mid = disarr[ 0 ] / disarr[ disarrsize / 2 ];
        double dismid_min = disarr[ disarrsize / 2 ] / disarr[ disarrsize - 2 ];

        if ( dismax_mid >= limit_maxmid || dismid_min >= limit_midmin )
        {
            curr_direct.setZero();
            return 0;
        }
    }
    else
    {
        double dismax_min = disarr[ 0 ] / disarr[ disarrsize - 2 ];
        if ( dismax_min >= limit_maxmin )
        {
            curr_direct.setZero();
            return 0;
        }
    }

    curr_direct << vx, vy, vz;
    curr_direct.normalize();
    return 1;
}

bool edge_jump_judge( const pcl::PointCloud< PointType > &pl, vector< orgtype > &types, uint i, Surround nor_dir )
{
    if ( nor_dir == 0 )
    {
        if ( types[ i - 1 ].range < blind || types[ i - 2 ].range < blind )
        {
            return false;
        }
    }
    else if ( nor_dir == 1 )
    {
        if ( types[ i + 1 ].range < blind || types[ i + 2 ].range < blind )
        {
            return false;
        }
    }
    double d1 = types[ i + nor_dir - 1 ].dista;
    double d2 = types[ i + 3 * nor_dir - 2 ].dista;
    double d;

    if ( d1 < d2 )
    {
        d = d1;
        d1 = d2;
        d2 = d;
    }

    d1 = sqrt( d1 );
    d2 = sqrt( d2 );

    if ( d1 > edgea * d2 || ( d1 - d2 ) > edgeb )
    {
        return false;
    }

    return true;
}
