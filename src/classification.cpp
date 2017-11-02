#include "classification.h"

using namespace std;
using namespace TooN;


#define C_TYPE_NO_CONTACT               "0"
#define C_TYPE_CONTACT_CONCORDE         "1"
#define C_TYPE_CONTACT_OPPOSITE         "2"
#define C_TYPE_CONTACT_DEVIATION_C       "3"
#define C_TYPE_CONTACT_DEVIATION_O       "4"

void load_param( double & p, double def, string name ) {
  ros::NodeHandle n_param("~");
  if( !n_param.getParam( name, p))
    p = def;
  cout << name << ": " << "\t" << p << endl;
}

void load_param( string & p, string def, string name ) {
  ros::NodeHandle n_param("~");
  if( !n_param.getParam( name, p))
    p = def;
  cout << name << ": " << "\t" << p << endl;
}

void load_param( int & p, int def, string name ) {
  ros::NodeHandle n_param("~");
  if( !n_param.getParam( name, p))
    p = def;
  cout << name << ": " << "\t" << p << endl;
}

void load_param( bool & p, bool def, string name ) {
  ros::NodeHandle n_param("~");
  if( !n_param.getParam( name, p))
    p = def;
  cout << name << ": " << "\t" << p << endl;
}

void load_param( int & p, bool def, string name ) {
  ros::NodeHandle n_param("~");
  if( !n_param.getParam( name, p))
    p = def;
  cout << name << ": " << "\t" << p << endl;
}

string get_c_name(int c) {

  //return _label_maps.at(c);
  /*
  if ( c == -1 )
    return "C_TYPE_NO_CLASSIFICATION";
  else if( c == 0 )
    return "C_TYPE_NO_CONTACT";
  if( c == 1 )
    return "C_TYPE_CONTACT_CONCORDE";
  if( c == 2 )
    return "C_TYPE_CONTACT_OPPOSITE";
  else if( c == 3 ) 
    return "C_TYPE_CONTACT_DEVIATION_C";
  else if( c == 4 ) 
    return "C_TYPE_CONTACT_DEVIATION_O";
  */
  return "";
}


NN_classification::NN_classification() {

  
  string train_file;
  string test_file;
  load_param(_ts_type, "ts_mini45", "ts_type");
  load_param(train_file, "train_dirforce_normal", "training_set");
  load_param(test_file, "test_dirforce_normal", "testing_set");
  load_param(_dimension, 2, "dimension");
  load_param(_classes, 4, "classes");
  load_param(_test_nn, true, "test_nn");
  load_param(_gen_tt_files, false, "gen_tt_files");
  load_param(_rate, 100, "rate");
  load_param(_hlayer, 10, "hlayer");
  load_param(_seq_samples, 10,  "seq_samples");
  load_param(_seq_samples_start, 50,  "seq_samples_start");
  load_param(_enable_service, true, "enable_service");

  _dir_path = ros::package::getPath("human_contact_classification");
  _dir_path += "/" + _ts_type + "/";


  _train_filename = _dir_path + train_file;
  _test_filename = _dir_path + test_file;
  _label_filename = _dir_path + "labels";

  cout << "Training Set filename: " << _train_filename << endl;
  cout << "Test Set filename: " << _test_filename << endl;

  _eef_p_start_direction_sub  = _nh.subscribe( "/iiwa/eef_p_start_direction", 0, &NN_classification::pdir_start_cb, this );
  
  //----Service is not working
  if(!_enable_service) {
    //Direzione pianificata
    /*
    _eef_p_direction_sub        = _nh.subscribe( "/iiwa/eef_p_direction", 0, &NN_classification::pdir_cb, this );  
    _eef_h_direction_sub        = _nh.subscribe( "/iiwa/eef_h_direction", 0, &NN_classification::hdir_cb, this );  
    _force_sub                  = _nh.subscribe( "/fts_data", 0, &NN_classification::force_cb, this );  
    _dist_from_traj_sub         = _nh.subscribe( "/iiwa/dist_from_traj", 0, &NN_classification::dist_cb, this);
    */
    _eef_p_direction_sub        = _nh.subscribe( "/pdir", 0, &NN_classification::pdir_cb, this );  
    _eef_h_direction_sub        = _nh.subscribe( "/hdir", 0, &NN_classification::hdir_cb, this );  
    _force_sub                  = _nh.subscribe( "/human_force", 0, &NN_classification::force_cb, this );  
    _dist_from_traj_sub         = _nh.subscribe( "/tdist", 0, &NN_classification::dist_cb, this);
    _class_pub                  = _nh.advertise< std_msgs::String > ("/nn_classification/class", 0);  
    
  }
  else {
    cout << "ADVERTISE SERVICE!" << endl;
    _classify_service_server = _nh.advertiseService("/human_contact_classification/classify", &NN_classification::srv_classification, this);

  }
  //---

  _angle_pub                  = _nh.advertise< std_msgs::Float64> ("/iiwa/shared/angle", 0);
  _pdir_start = Zeros;
  _start_point_classification = false;

  
  _confusion_Matrix.resize( _classes );
  for(int i=0; i<_classes; i++) {
    _confusion_Matrix[i].resize( _classes );
  }

  for(int i=0; i<_classes; i++) {
    for(int j=0; j<_classes; j++) {
      _confusion_Matrix[i][j] = 0;
    }
  }  

  _sample_in_class.resize( _classes );

}

NN_classification::~NN_classification() { 
  cvReleaseMat( &_training_data );
  cvReleaseMat( &_training_classifications );
  cvReleaseMat( &_testing_data );
  cvReleaseMat( &_testing_classifications );
  cvReleaseMat( &_classificationResult);
}

void NN_classification::pdir_start_cb( geometry_msgs::Vector3 pdir_start ) {
  _pdir_start = makeVector( pdir_start.x, pdir_start.y, pdir_start.z );
  _start_point_classification = true;

  //cout << "_pdir_start: " << _pdir_start << " / " << norm(_pdir_start) << endl;

}

void NN_classification::force_cb( geometry_msgs::Wrench wrench_) {
  _force = makeVector( wrench_.force.x, wrench_.force.y, wrench_.force.z );
}

void NN_classification::pdir_cb( geometry_msgs::Vector3 pdir ) {
  _pdir = makeVector( pdir.x, pdir.y, pdir.z );
}

void NN_classification::hdir_cb( geometry_msgs::Vector3 hdir ) {
  _hdir = makeVector( hdir.x, hdir.y, hdir.z ); 
}

void NN_classification::dist_cb( std_msgs::Float64 dist_ ) {
  _dist = dist_.data;
}


bool read_data_from_csv(const char* filename, CvMat* data, CvMat* classes,
  
  int n_samples, int attributes_per_sample ) {

  int classlabel; // the class label

  FILE* f = fopen( filename, "r" );
  if( !f )  {
    printf("ERROR: cannot read file %s\n",  filename);
    return false;
  }

  for(int line = 0; line < n_samples; line++) {    

    for(int attribute = 0; attribute < (attributes_per_sample + 1); attribute++) {
      if (attribute < attributes_per_sample) {    
        //printf("%f\n", f );
        fscanf(f, "%f,", &(CV_MAT_ELEM  (*data, float, line, attribute)));
        //cout << "ELEM: " << (CV_MAT_ELEM  (*data, float, line, attribute)) << " ";
      }
      else if (attribute == attributes_per_sample) {          
        fscanf(f, "%i,", &classlabel);
        CV_MAT_ELEM(*classes, float, line, classlabel) = 1.0; // TODO: Change classlabel with an id (now is an index, must start from 0!)
        //cout << "[" << classlabel << "]" << endl;
      }      
    }
    //cout << endl;
  }
  fclose(f);

  return true;
}


bool NN_classification::train() {

  _training_data = cvCreateMat( _train_sample_num, _dimension, CV_32FC1);
  _training_classifications = cvCreateMat(_train_sample_num, _classes, CV_32FC1);
  cvZero(_training_classifications);
  
  if( !read_data_from_csv(helper::string2char( _train_filename ), 
    _training_data, _training_classifications, _train_sample_num, _dimension ))
    return false;

  
  _testing_data = cvCreateMat(_test_sample_num, _dimension, CV_32FC1);
  _testing_classifications = cvCreateMat(_test_sample_num, _classes, CV_32FC1);
  cvZero(_testing_classifications);

  if( !read_data_from_csv(helper::string2char( _test_filename ), 
    _testing_data, _testing_classifications, _test_sample_num, _dimension ))
    return false;
  
  _classificationResult = cvCreateMat(1, _classes, CV_32FC1);
  
  
  int layers_d[] = { _dimension, _hlayer,  _classes};   
  CvMat* layers = cvCreateMatHeader(1,3,CV_32SC1);
  cvInitMatHeader(layers, 1,3,CV_32SC1, layers_d);

  // create the network using a sigmoid function with alpha and beta
  // parameters 0.6 and 1 specified respectively (refer to manual)
  _nnetwork = new CvANN_MLP;
  _nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

  CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams(
          // terminate the training after either 1000
          // iterations or a very small change in the
          // network wieghts below the specified value
          cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
          // use backpropogation for training
          CvANN_MLP_TrainParams::BACKPROP,
          // co-efficents for backpropogation training
          // (refer to manual)
          0.1,
          0.1);
  
  int iterations = _nnetwork->train(_training_data, _training_classifications, NULL, NULL, params);

  return true;
}


void NN_classification::nn_testing() {

  CvMat test_sample;
  int correct_class = 0;
  int wrong_class = 0;
  int false_positives [_classes];
  CvPoint max_loc = {0,0};


  vector<int> conf_vector_c1;
  vector<int> conf_vector_c2;
  vector<int> conf_vector_c3;
  vector<int> conf_vector_c4;

  for( int i=0; i<_classes; i++ )
    false_positives[i] = 0;
  
  for (int tsample = 0; tsample < _test_sample_num; tsample++) {
      
    // extract a row from the testing matrix
    cvGetRow(_testing_data, &test_sample, tsample );

    _nnetwork->predict(&test_sample, _classificationResult);

    // The NN gives out a vector of probabilities for each class
    // We take the class with the highest "probability"
    // for simplicity (but we really should also check separation
    // of the different "probabilities" in this vector - what if
    // two classes have very similar values ?)
    cvMinMaxLoc(_classificationResult, 0, 0, 0, &max_loc, 0 );
    //cout << "TEST SAMPLE: " << CV_MAT_ELEM(test_sample, float, 0, 3) << endl;
    /*
    cout << CV_MAT_ELEM(test_sample, float, 0, 0) << " " <<     
    CV_MAT_ELEM(test_sample, float, 0, 1) <<  " " << 
    CV_MAT_ELEM(test_sample, float, 0, 2) <<  " " << /*
    CV_MAT_ELEM(test_sample, float, 0, 3) <<  " " << 
    CV_MAT_ELEM(test_sample, float, 0, 4) <<  " " << 
    CV_MAT_ELEM(test_sample, float, 0, 5) <<  " " << endl; */

    /*
    int rc2 = -1;
      for( int i =0 ; i<_classes; i++ ) {
        if( CV_MAT_ELEM(*_testing_classifications, float, tsample, i) == 1 )
          rc2 = i;
      }
      
    printf("Testing Sample %i [%i] -> class result (digit %d)\n", tsample, rc2, max_loc.x);
    */
    /**/

    // if the corresponding location in the testing classifications
    // is not "1" (i.e. this is the correct class) then record this
    if (!(CV_MAT_ELEM(*_testing_classifications, float, tsample, max_loc.x))) {
      // if they differ more than floating point error => wrong class
      wrong_class++;
      false_positives[(int) max_loc.x]++;

      //cout << "wrong!" << endl;
      int rc = -1;
      for( int i =0 ; i<_classes; i++ ) {
        if( CV_MAT_ELEM(*_testing_classifications, float, tsample, i) == 1 )
          rc = i;
      }
      _sample_in_class[rc]++;
      //cout << "Samples: " << _sample_in_class[rc] << endl;
      /*  << " "
        CV_MAT_ELEM(*_testing_classifications, float, tsample, 1)  << " "
          CV_MAT_ELEM(*_testing_classifications, float, tsample, 2)  << endl;
      */
      //find correct class!
      //_confusion_Matrix[(int) max_loc.x][rc]++;
      _confusion_Matrix[rc][(int) max_loc.x]++;
      //cout << "IN: " << (int) max_loc.x<< " " << rc << " (" << _confusion_Matrix[(int) max_loc.x][rc] << ")" << endl;

    }
    else {
      _confusion_Matrix[(int) max_loc.x][(int) max_loc.x]++;
      _sample_in_class[(int) max_loc.x]++;

      // otherwise correct
      correct_class++;
    }
  }

  printf( "\nResults on the testing database: \n" 
          "\tCorrect classification: %d (%g%%)\n" 
          "\tWrong classifications: %d (%g%%)\n",
          correct_class, (double) correct_class*100/_test_sample_num,
          wrong_class, (double) wrong_class*100/_test_sample_num);

  for (int i = 0; i < _classes; i++) {
    printf( "\tClass (digit %d) false postives  %d (%g%%)\n", i, false_positives[i],
         (double) false_positives[i]*100/_test_sample_num);
  }
  cout << "-------------Confusion matrix---------------" << endl;
  for (int i=0; i<_confusion_Matrix.size(); i++ ) {
    for (int j=0; j<_confusion_Matrix.size(); j++ ) {
      cout << (double)_confusion_Matrix[i][j]*100/_sample_in_class[i] << "%\t";
    }
    cout << "\t" << _sample_in_class[i] << endl;
  }
  cout << "-------------------------------------------" << endl;
}


void NN_classification::gen_tt_files() {

  ros::Rate r(_rate);

  string force_file_name;
  string direction_file_name;
  string complete_file_name;
  string norm_force_file_name;
  string dirforce_file_name;
  string dirforcedist_file_name;

  string label_filename;

  string line;
  string contact;
  
  cout << "Insert data type: " << endl;
  cout << "1 - TRAIN" << endl;
  cout << "2 - TEST" << endl;
  getline(cin, line); 

  string prefix = "";
  if( line == "1") 
    prefix = "train_";
  else if( line == "2" )
    prefix = "test_";

  cout << "Insert class: " << endl;
  cout << "1 - no contact" << endl;
  cout << "2 - contact_concorde" << endl;
  cout << "3 - contact_opposite" << endl;
  cout << "4 - contact_deviation_compliant" << endl;
  cout << "5 - contact_deviation_opposite" << endl;
  
  getline(cin, line);
  contact = line;

  if( line == "1" )
    contact = "C_TYPE_NO_CONTACT";
  else if( line == "2" )
    contact = "C_TYPE_CONTACT_CONCORDE";
  else if ( line == "3" )
    contact = "C_TYPE_CONTACT_OPPOSITE";
  else if ( line == "4" )
    contact = "C_TYPE_CONTACT_DEVIATION_C";
  else if ( line == "5" )
    contact = "C_TYPE_CONTACT_DEVIATION_O";
  
  dirforcedist_file_name = _dir_path + prefix + "dirforcedist_" + contact;  

  _dirforcedist_file.open( helper::string2char( dirforcedist_file_name ), std::ofstream::out | std::ofstream::app);
  /*
  _label_file.open( helper::string2char( _label_filename ), std::ofstream::out | std::ofstream::app);
  cout << "Insert LABEL for this contact type: " << endl;
  cout << "1 - C_TYPE_NO_CONTACT" << endl;
  cout << "2 - C_TYPE_CONTACT_CONCORDE" << endl;
  cout << "3 - C_TYPE_CONTACT_OPPOSITE" << endl;
  cout << "4 - C_TYPE_CONTACT_DEVIATION_C" << endl;
  cout << "5 - C_TYPE_CONTACT_DEVIATION_O" << endl;
  getline(cin, line); 

  if( line == "1" )
    _label_file << contact << " " << "C_TYPE_NO_CONTACT" << endl;
  else if( line == "2" )
    _label_file << contact << " " << "C_TYPE_CONTACT_CONCORDE" << endl;
  else if ( line == "3" )
    _label_file << contact << " " << "C_TYPE_CONTACT_OPPOSITE" << endl;
  else if ( line == "4" )
    _label_file << contact << " " << "C_TYPE_CONTACT_DEVIATION_C" << endl;
  else if ( line == "5" )
    _label_file << contact << " " << "C_TYPE_CONTACT_DEVIATION_O" << endl;
  */

  float angle;
  Vector<3> axis;
  
  //exit(1);

  Vector<3> nforce;
  while( ros::ok() ) {

    if( norm( _pdir) && norm( _hdir ) > 0.0 && norm(_force) > 0.0 ) { 
      angle = atan2( norm( _pdir ^ _hdir ), _pdir * _hdir);
      nforce = _force;
      nforce = nforce / norm(nforce);
      _dirforcedist_file << norm(_force) << " " << angle <<  " " << _dist << endl;
    }
    else {
      angle = 0.0;
    }
    r.sleep();
  }
}



string NN_classification::classify(float angle, float nforce, float _dist) {
  
  CvMat *c_sample = cvCreateMat(1, _dimension, CV_32FC1);;
  CvPoint max_loc = {0,0};

  CV_MAT_ELEM(*c_sample, float, 0, 0) = norm( _force );
  CV_MAT_ELEM(*c_sample, float, 0, 1) = angle;
  CV_MAT_ELEM(*c_sample, float, 0, 2) = _dist;

  _nnetwork->predict( c_sample, _classificationResult);
  cvMinMaxLoc(_classificationResult, 0, 0, 0, &max_loc, 0 );      

  return _label_maps.find((int)max_loc.x)->second; 
}


void NN_classification::classification() {

  ros::Rate r(_rate );
  
  float angle;
  Vector<3> nforce;
  CvMat *c_sample = cvCreateMat(1, _dimension, CV_32FC1);;
  CvPoint max_loc = {0,0};
  string c = "";
  string p_c = "";
  bool first = true;
  bool first_start = true;
  int c_seq = 0;
  std_msgs::String c_data;
  std_msgs::String c_data_start;
  int c_seq_zero = 0;

  int c_seq_start = 0;
  int c_seq_zero_start = 0;
  string c_start = "";
  string p_c_start = "";
  std_msgs::Float64 angle_d;

  while( ros::ok() ) {

    if( norm( _pdir) > 0.0 && norm( _hdir ) > 0.0 && norm(_force) > 0.0 ) { 
      
      angle = atan2( norm( _pdir ^ _hdir ), _pdir * _hdir);    
      c = classify( angle, norm( _force ), _dist );
      
      if ( !first ) {
        if ( p_c == c ) {
          c_seq++;
          c_seq_zero = 0;          
        }
        else {
          if( c_seq_zero++ > 5 )
            c_seq = 0;
        }
      }
      else
        c_data.data = get_c_name( -1 );

      if ( c_seq > _seq_samples) {    
        c_data.data = c;
      }
      else {
        c_data.data = "C_TYPE_NO_CONTACT";      
      }
      
      c_data.data = c;
      cvMinMaxLoc(_classificationResult, 0, 0, 0, &max_loc, 0 );      
      
      first = false;
    }
    else {        
      c_data.data = "C_TYPE_NO_CONTACT";
      angle = 0;        
    }

    p_c = c;
  
      angle_d.data = angle;
    _angle_pub.publish( angle_d  );
    _class_pub.publish( c_data );  

    r.sleep();
  }


}

bool NN_classification::srv_classification(human_contact_classification::classify::Request &req, human_contact_classification::classify::Response &res ) {

  float angle;
  Vector<3> nforce;
  CvPoint max_loc = {0,0};
  CvMat *c_sample = cvCreateMat(1, _dimension, CV_32FC1);;
  string c = "";
  std_msgs::String c_data;  
  std_msgs::Float64 angle_d;

  //---Set data  
  _pdir   = makeVector(req.p_dir.x, req.p_dir.y, req.p_dir.z);
  _hdir   = makeVector(req.h_dir.x, req.h_dir.y, req.h_dir.z);
  _force  = makeVector( req.wrench.force.x, req.wrench.force.y, req.wrench.force.z);
  _dist   = req.tdist.data;
  //---

  //cout << norm( _pdir) << " " <<  norm( _hdir ) << " " <<  norm(_force) << endl;
  if( norm( _pdir) > 0.0 && norm( _hdir ) > 0.0 && norm(_force) > 3.0 ) { 
      
    angle = atan2( norm( _pdir ^ _hdir ), _pdir * _hdir);    
    c = classify( angle, norm( _force ), _dist );
    cvMinMaxLoc(_classificationResult, 0, 0, 0, &max_loc, 0 );        
    //first = false;
  }
  else {
    //c_data.data = get_c_name( -1 );
    //angle = 0;
    c = "C_TYPE_NO_CONTACT";
  }
  //cout << "Classification: " << c << endl;
  //p_c = c;
  res.classification = c;

  //else 
  //  c_data.data = "C_TYPE_NO_CONTACT";
  angle_d.data = angle;
  _angle_pub.publish( angle_d  );
  //_class_pub.publish( c_data );  
 // _start_class_pub.publish( c_data_start );  

  return true;
}


void NN_classification::run() {
  
  if ( _gen_tt_files)
    boost::thread gen_tt_files_t( &NN_classification::gen_tt_files, this );
  else {
    string line;

    vector<string> classes;
    //---Hardcoded classname
    classes.push_back("C_TYPE_CONTACT_CONCORDE");
    _label_maps.insert ( std::pair<int,string>(0, "C_TYPE_CONTACT_CONCORDE") );
    classes.push_back("C_TYPE_CONTACT_OPPOSITE");
    _label_maps.insert ( std::pair<int,string>(1, "C_TYPE_CONTACT_OPPOSITE") );
    classes.push_back("C_TYPE_CONTACT_DEVIATION_C");
    _label_maps.insert ( std::pair<int,string>(2, "C_TYPE_CONTACT_DEVIATION_C") );
    classes.push_back("C_TYPE_CONTACT_DEVIATION_O");
    _label_maps.insert ( std::pair<int,string>(3, "C_TYPE_CONTACT_DEVIATION_O") );  
    //---


    std::ofstream train_file_to_write ( helper::string2char( _train_filename ) );
    for( int i=0; i<classes.size(); i++ ) {
      string dirforcedist_file_name = _dir_path + "train_" + "dirforcedist_" + classes[i];  
      ifstream datafile;
      datafile.open( helper::string2char( dirforcedist_file_name ));
      while (std::getline(datafile, line)) {
        train_file_to_write <<  line << " " << i << endl;
      }
    }
  
    std::ofstream test_file_to_write ( helper::string2char( _test_filename ) );
    for( int i=0; i<classes.size(); i++ ) {
      string dirforcedist_file_name = _dir_path + "test_" + "dirforcedist_" + classes[i];  
      ifstream datafile;
      datafile.open( helper::string2char( dirforcedist_file_name ));
      while (std::getline(datafile, line)) {
        test_file_to_write << line << " " << i << endl;
      }
    }

    _train_sample_num = _test_sample_num = 0;
      
    std::ifstream train_file ( helper::string2char( _train_filename ) );
    while (std::getline(train_file, line))
      ++_train_sample_num;
    std::cout << "Number of lines in training file: " << _train_sample_num << endl;

    cout << "pre testfile open" << endl;
    std::ifstream test_file ( helper::string2char( _test_filename ) );
    while (std::getline(test_file, line))
      ++_test_sample_num;
    std::cout << "Number of lines in testing file: " << _test_sample_num << endl;

    cout << "Start training..." << endl;
    bool train_ok = train();
    if( train_ok ) 
      cout << "...NN Trained!" << endl << endl; 


    cout << "Start testing..." << endl;
    if( _test_nn ) {
      nn_testing();
    }
    cout << "...Test complete!" << endl;

    if( !_enable_service)
      boost::thread classification_t( &NN_classification::classification, this );
  }
  ros::spin();
}


int main( int argc, char** argv ) {

  ros::init( argc, argv, "nn_classification");
  NN_classification nn;
  nn.run();

  return 0;
}

