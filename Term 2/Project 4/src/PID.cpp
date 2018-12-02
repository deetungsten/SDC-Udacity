#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    PID::Kp = Kp;
    PID::Ki = Ki;
    PID::Kd = Kd;
    i_error_temp = 0;
    dt = 0;
    previous_cte = 0;
}

void PID::UpdateError(double cte ,double dt) {
    p_error = Kp *cte;
    i_error_temp = (cte*dt + 0.8*i_error_temp); //dampen the steady state error
    i_error = Ki*i_error_temp;
    d_error = (cte-previous_cte)*Kd/dt;
    
    previous_cte = cte;
    //total_error += cte;

}

double PID::TotalError() {
    return -(p_error + i_error + d_error);
}

