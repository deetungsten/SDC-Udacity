#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x = VectorXd(5);

  // initial covariance matrix
  P = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd = .375;

  // Laser measurement noise standard deviation position1 in m
  std_laspx = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  n_x = 5;

  n_aug = 7;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    
    // first measurement

    float px;
    float py;
    float vx;
    float vy;
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
     
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];
        
      px = rho * cos(phi);
      py = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
    
      
      x << px, py, vx, vy, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
     
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      
      x << px, py, 0, 0, 0;
     
    }
    previous_timestamp_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
	}

	double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = meas_package.timestamp_;


			   
	Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
    
  } else {
    // Laser updates
    UpdateLidar(meas_package); 
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_out;  
  GenerateSigmaPoints(&Xsig_out);
  AugmentedSigmaPoints(&Xsig_out);
  SigmaPointPrediction(delta_t,&Xsig_out);
  PredictMeanAndCovariance(&Xsig_out);
}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //define spreading parameter
  double lambda = 3 - n_x;

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

  //calculate square root of P
  MatrixXd A = P.llt().matrixL();

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //set first column of sigma point matrix
  Xsig.col(0)  = x;

  //set remaining sigma points
  for (int i = 0; i < n_x; i++)
  {
    Xsig.col(i+1)     = x + sqrt(lambda+n_x) * A.col(i);
    Xsig.col(i+1+n_x) = x - sqrt(lambda+n_x) * A.col(i);
  }

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  //write result
  *Xsig_out = Xsig;

}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {


  //define spreading parameter
  double lambda = 3 - n_aug;

  //set example state
  //VectorXd x = VectorXd(n_x);

  //create example covariance matrix
  //MatrixXd P = MatrixXd(n_x, n_x);

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/
 
  //create augmented mean state
  x_aug.head(5) = x;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P;
  P_aug(5,5) = std_a*std_a;
  P_aug(6,6) = std_yawdd*std_yawdd;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda+n_aug) * L.col(i);
    Xsig_aug.col(i+1+n_aug) = x_aug - sqrt(lambda+n_aug) * L.col(i);
  }
  
/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;


}


void UKF::SigmaPointPrediction( double delta_t,MatrixXd* Xsig_out) {


  //create example sigma point matrix
  Xsig_aug = *Xsig_out;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //predict sigma points
  for (int i = 0; i< 2*n_aug+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;

}

void UKF::PredictMeanAndCovariance(MatrixXd* Xsig_out) {

  Xsig_pred = *Xsig_out;
  //define spreading parameter
  double lambda = 3 - n_aug;

  //create example matrix with predicted sigma points
  //MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  //create vector for weights
  VectorXd weights = VectorXd(2*n_aug+1);
  
  //create vector for predicted state
  //VectorXd x = VectorXd(n_x);

  //create covariance matrix for prediction
  //MatrixXd P = MatrixXd(n_x, n_x);


/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  // set weights
  double weight_0 = lambda/(lambda+n_aug);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug+lambda);
    weights(i) = weight;
  }

  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points
    x = x+ weights(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P = P + weights(i) * x_diff * x_diff.transpose() ;
  }


/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Predicted state" << std::endl;
  //std::cout << x << std::endl;
  //std::cout << "Predicted covariance matrix" << std::endl;
  //std::cout << P << std::endl;

  //write result
}

void UKF::PredictMeasurement(int n_z) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  //int n_z = 3;

  //define spreading parameter
  double lambda = 3 - n_aug;


  //radar measurement noise standard deviation radius in m
  //double std_radr = 0.3;

  //radar measurement noise standard deviation angle in rad
  //double std_radphi = 0.0175;

  //radar measurement noise standard deviation radius change in m/s
  //double std_radrd = 0.1;

  //create example matrix with predicted sigma points
  //MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  Zsig.fill(0.0);
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v  = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    if(n_z==3){
    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
    else{
    Zsig(0,i) = p_x;                        //r
    Zsig(1,i) = p_y;                                 //phi
    
    }
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);

  if(n_z==3){
  R <<    std_radr*std_radr, 0, 0,
          0, std_radphi*std_radphi, 0,
          0, 0,std_radrd*std_radrd;
  }
  else{
  R <<    pow(std_laspx,2), 0,
          0, pow(std_laspy,2);

  }
  S = S + R;

  
/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  //*z_out = z_pred;
  //*S_out = S;
}




void UKF::UpdateState(int n_z, VectorXd z) {

//#########################################

  //set measurement dimension, radar can measure r, phi, and r_dot
  //int n_z = 3;

  //define spreading parameter
  double lambda = 3 - n_aug;

  VectorXd weights = VectorXd(2*n_aug+1);
  double weight_0 = lambda/(lambda+n_aug);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug+lambda);
    weights(i) = weight;
  }

  //radar measurement noise standard deviation radius in m
  //double std_radr = 0.3;

  //radar measurement noise standard deviation angle in rad
  //double std_radphi = 0.0175;

  //radar measurement noise standard deviation radius change in m/s
  //double std_radrd = 0.1;

  //create example matrix with predicted sigma points
  //MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  //Zsig.fill(0.0);
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v  = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    if(n_z==3){
    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
    else{
    Zsig(0,i) = p_x;                        //r
    Zsig(1,i) = p_y;                                 //phi
    
    }
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);

  if(n_z==3){
  R <<    std_radr*std_radr, 0, 0,
          0, std_radphi*std_radphi, 0,
          0, 0,std_radrd*std_radrd;
  }
  else{
  R <<    pow(std_laspx,2), 0,
          0, pow(std_laspy,2);

  }
  S = S + R;

  
/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  //*z_out = z_pred;
  //*S_out = S;




  //#################################################

  //set measurement dimension, radar can measure r, phi, and r_dot

  //define spreading parameter
  //#####double lambda = 3 - n_aug;


  //create example matrix with predicted sigma points
  //MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  //create example vector for predicted state mean
  //VectorXd x = VectorXd(n_x);

  //create example matrix for predicted state covariance
  //MatrixXd P = MatrixXd(n_x,n_x);


  //create example matrix with sigma points in measurement space
  //MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

  //create example vector for mean predicted measurement
  //VectorXd z_pred = VectorXd(n_z);


  //create example matrix for predicted measurement covariance
  //MatrixXd S = MatrixXd(n_z,n_z);

  //create example vector for incoming radar measurement
  //VectorXd z = VectorXd(n_z);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K*S*K.transpose();

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Updated state x: " << std::endl << x << std::endl;
  //std::cout << "Updated state covariance P: " << std::endl << P << std::endl;

  if(n_z==3){
    NIS_radar = z_diff.transpose() * S.transpose() * z_diff;
  }
  else{
    NIS_lidar = z_diff.transpose() * S.transpose() * z_diff;
  }

}



/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //PredictMeasurement(2);

  VectorXd z = VectorXd(2);
  //double measurement_px = meas_package.raw_measurements_[0];
  //double measurement_py = meas_package.raw_measurements_[1];

  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  UpdateState(2,z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //PredictMeasurement(3);

  VectorXd z = VectorXd(3);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1],meas_package.raw_measurements_[2];
  UpdateState(3,z);
}



