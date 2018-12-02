/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 blahh
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//cout << "START";
	default_random_engine gen;
	//double std_x, std_y, std_theta; // Standard deviations for x, y, and psi

	num_particles = 100;
	// TODO: Set standard deviations for x, y, and psi
	 

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);


	weights.resize(num_particles);
	particles.resize(num_particles);

	for (int i = 0; i < num_particles; ++i) {
		 weights[i] = 1;		
		 particles[i].x = dist_x(gen);
		 particles[i].y = dist_y(gen);
		 particles[i].theta = dist_theta(gen);	 
		 particles[i].weight = 1;
		 particles[i].id = i;
		 // Print your samples to the terminal.
		 //cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << endl;
	}
	//cout << "DONE";
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	double x_0;
	double y_0;
	double theta_0;

	double x_f;
	double y_f;
	double theta_f;

	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);


	for (int i = 0; i < num_particles; ++i) {
		x_0 = particles[i].x;
		y_0 = particles[i].y;
		theta_0 = particles[i].theta;

		if(abs(yaw_rate) < 1e-5) {

			x_f = x_0 + velocity * delta_t * cos(theta_0);
			y_f = y_0 + velocity * delta_t * sin(theta_0);
			theta_f = theta_0;
			//cout << "straight\n";
			//cout << yaw_rate;
	
		}
		else{
			x_f = x_0 + (velocity/yaw_rate) * (sin(theta_0+(yaw_rate*delta_t)) - sin(theta_0));
			y_f = y_0 + (velocity/yaw_rate) * (-cos(theta_0+(yaw_rate*delta_t)) + cos(theta_0));
			theta_f = theta_0 + yaw_rate * delta_t;
			//cout << "curved\n";
			//cout << yaw_rate;
			//cout << x_f;
		}

		particles[i].x = x_f + noise_x(gen);
		particles[i].y = y_f + noise_y(gen);
		particles[i].theta = theta_f + noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	


	double landmark_x;
	double landmark_y;
	int min_pos;
	double nearest_distance; 
	

	

	for (int n = 0; n < num_particles; ++n) {

		std::vector<LandmarkObs>transform_observations;
		
		long double multi_gaus = 1.0;
		

		for(int i=0; i < observations.size(); ++i){
			LandmarkObs transform_obs;
			transform_obs.x = (observations[i].x * cos(particles[n].theta)) - (observations[i].y * sin(particles[n].theta)) + particles[n].x;
			transform_obs.y = (observations[i].x * sin(particles[n].theta)) + (observations[i].y * cos(particles[n].theta)) + particles[n].y;
			transform_observations.push_back(transform_obs);
		}

		particles[n].weight = 1.0;
		for(int m=0; m < transform_observations.size(); ++m){
			double map_distance_temp;
			std::vector<double>map_distance;
			for(int j=0; j < map_landmarks.landmark_list.size(); ++j){
				landmark_x = map_landmarks.landmark_list[j].x_f;	
				landmark_y = map_landmarks.landmark_list[j].y_f;
	
				map_distance_temp= sqrt(pow(transform_observations[m].x - landmark_x,2.0) + pow(transform_observations[m].y - landmark_y,2.0));
				map_distance.push_back(map_distance_temp);
	


			
			
			}


				min_pos = distance(map_distance.begin(),min_element(map_distance.begin(),map_distance.end()));
				double closest_x = transform_observations[m].x;
				double closest_y = transform_observations[m].y;

				double landmark_distance_x = map_landmarks.landmark_list[min_pos].x_f;
				double landmark_distance_y = map_landmarks.landmark_list[min_pos].y_f;
				
				double sigma_x = std_landmark[0];
				double sigma_y = std_landmark[1];

				double pow_diff_x = pow((closest_x - landmark_distance_x),2.0)/pow(sigma_x,2.0);
				double pow_diff_y = pow((closest_y - landmark_distance_y),2.0)/pow(sigma_y,2.0);

				//long double measurement_prob = 1/(2*M_PI*sigma_x*sigma_y) * exp((-1/2)*(pow(closest_x-landmark_distance_x,2.0)/(sigma_x*sigma_x)+pow(closest_y-landmark_distance_y,2.0)/(sigma_y*sigma_y)));
				long double measurement_prob = (1/(2*M_PI*sigma_x*sigma_y)) * exp((-0.5)*(pow_diff_x+pow_diff_y));
				//cout << measurement_prob;
				//cout << measurement_prob;
				
				multi_gaus *= measurement_prob;

			
				
			}
		particles[n].weight = multi_gaus;
		//cout << min_pos;
		weights[n] = multi_gaus;
			}
		}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> discrete(weights.begin(),weights.end());

	std::vector<Particle> new_particles;
	std::vector<double> new_weights;

	for(int i=0; i < num_particles; ++i){
		new_particles.push_back(particles[discrete(gen)]);

		//new_weights.push_back(new_particles.weight);

	}

	particles = new_particles;
	//weights = new_weights;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
