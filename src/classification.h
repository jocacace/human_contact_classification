#include "ros/ros.h"
#include "TooN/TooN.h"
#include "robohelper/robohelper.hpp"
#include "sensor_msgs/JointState.h"
#include "boost/thread.hpp"
#include "geometry_msgs/Wrench.h"
#include "sensor_msgs/JointState.h"
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include "geometry_msgs/WrenchStamped.h"
#include "iiwa_kinematic/LBRiiwa.h"
#include "TooN/LU.h"
#include "6d_motion_planner/6d_motion_planner.h"
#include "iiwa_msgs/JointPosition.h"
#include "visualization_msgs/Marker.h"
#include <iostream>
#include "std_msgs/Float64.h"
#include "std_msgs/Bool.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Twist.h"
#include "std_msgs/Int32.h"
#include <ros/package.h>
#include "std_msgs/String.h" 
#include "human_contact_classification/classify.h"
#include <iostream>

#include "cv.h"       // opencv general include file
#include <opencv2/ml.hpp>     // opencv machine learning include file
#include <stdio.h>

using namespace std;

class NN_classification {
  public:
    NN_classification();
    ~NN_classification();
		void run();
		bool train();
		void nn_testing();
		void hdir_cb( geometry_msgs::Vector3 hdir );
		void pdir_cb( geometry_msgs::Vector3 pdir );
		void pdir_start_cb( geometry_msgs::Vector3 pdir );
		void force_cb( geometry_msgs::Wrench force_);
		void dist_cb( std_msgs::Float64 dist_ );
		void gen_tt_files();
		void classification();
		string classify(float angle, float nforce, float _dist);
		bool srv_classification(human_contact_classification::classify::Request& req, human_contact_classification::classify::Response& res );

  private:
    ros::NodeHandle _nh;    
    string _train_filename;
    string _test_filename;
    string _dir_path;
    string _ts_type;
    int _dimension;
    int _classes;
    bool _test_nn;
    bool _gen_tt_files;
    int _hlayer;
    int _rate;
    int _seq_samples;
    int _seq_samples_start;
		Vector<3> _pdir;
		Vector<3> _pdir_start;
		Vector<3> _hdir;
		Vector<3> _force;
		float _dist;
	
		ros::Publisher  _class_pub;
		ros::Publisher  _start_class_pub;	
		ros::Publisher  _angle_pub;	
    ros::Subscriber _eef_p_direction_sub;
		ros::Subscriber _eef_h_direction_sub;
		ros::Subscriber _eef_p_start_direction_sub;
		ros::Subscriber _force_sub;
		ros::Subscriber _dist_from_traj_sub;
		
		std::ofstream _norm_force_file;	  
	  std::ofstream _force_file;
	  std::ofstream _direction_file;
	  std::ofstream _complete_file;
	  std::ofstream _dirforce_file;
	  std::ofstream _dirforcedist_file;
		std::ofstream _label_file;

	  std::map<int, string> _label_maps;
	  string _label_filename;
	  bool 	 _enable_service;

    int _train_sample_num;
    int _test_sample_num;

		CvMat* _training_data;
		CvMat* _training_classifications; 
		CvMat* _testing_data;
		CvMat* _testing_classifications;	
		CvMat* _classificationResult;

		bool _start_point_classification;

		
		cv::Ptr<cv::ml::ANN_MLP> _nnetwork;
		vector < vector< int > > _confusion_Matrix;
		vector < int > _sample_in_class;

		ros::ServiceServer _classify_service_server;
};
