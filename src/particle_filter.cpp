/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

// The number of particles that will be used by the filter.
static const int NUM_PARTICLES = 1000;

// A small convenience.
static const double TWO_PI = 2.0 * M_PI;

// To avoid division by zero errors and numerical instability around zero, consider small numbers
// less than this to be zero.
static const double EPSILON = 1e-3;

// A pseudo random number generator that will be used to sample random distributions.
static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  std::normal_distribution<double> x_dist(x, std[0]);
  std::normal_distribution<double> y_dist(y, std[1]);
  std::normal_distribution<double> theta_dist(theta, std[2]);

  num_particles = NUM_PARTICLES;
  for (int i = 0; i < num_particles; ++i) {
    Particle particle = {
      .id = i,
      .x = x_dist(gen),
      .y = y_dist(gen),
      .theta = theta_dist(gen),
      .weight = 1.0
    };
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  std::normal_distribution<double> x_noise(0, std_pos[0]);
  std::normal_distribution<double> y_noise(0, std_pos[1]);
  std::normal_distribution<double> theta_noise(0, std_pos[2]);
  
  for (std::vector<Particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
    // Apply the motion model to predict new particle state. If the yaw rate is too small,
    // then assume the particles continue straight ahead with no change in yaw.
    double next_theta = it->theta + yaw_rate * delta_t;
    if (fabs(yaw_rate) < EPSILON) {
      it->x += velocity * cos(it->theta) * delta_t;
      it->y += velocity * sin(it->theta) * delta_t;
    } else {
      double r = velocity / yaw_rate;
      it->x += r * (sin(next_theta) - sin(it->theta));
      it->y += r * (cos(it->theta) - cos(next_theta));
    }
    it->x += x_noise(gen);
    it->y += y_noise(gen);
    it->theta = next_theta + theta_noise(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (std::vector<LandmarkObs>::iterator it_obs = observations.begin(); it_obs != observations.end(); ++ it_obs) {
    int nearest_id = -1;
    double nearest_distance = std::numeric_limits<double>::max();
    for (std::vector<LandmarkObs>::const_iterator it_pred = predicted.begin(); it_pred != predicted.end(); ++it_pred) {
      double distance = dist(it_obs->x, it_obs->y, it_pred->x, it_pred->y);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_id = it_pred->id;
      }
    }
    it_obs->id = nearest_id;
  }
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
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  const double std_x = std_landmark[0];
  const double std_y = std_landmark[1];
      
  for (std::vector<Particle>::iterator p_it = particles.begin(); p_it != particles.end(); ++p_it) {
    // Create vector of predicted locations of landmarks that are within sensor range of the particle.
    // Limiting the number of landmarks considered will improve the efficiency of the nearest neighbor
    // search.
    std::vector<LandmarkObs> predicted;
    for (std::vector<Map::single_landmark_s>::const_iterator ml_it = map_landmarks.landmark_list.begin(); ml_it != map_landmarks.landmark_list.end(); ++ml_it) {
      if (dist(p_it->x, p_it->y, ml_it->x_f, ml_it->y_f) <= sensor_range) {
        LandmarkObs obs = {
          .id = ml_it->id_i,
          .x = ml_it->x_f,
          .y = ml_it->y_f
        };
        predicted.push_back(obs);
      }
    }
    
    // Transform the sensor observations into map space.
    std::vector<LandmarkObs> observed;
    for (std::vector<LandmarkObs>::const_iterator o_it = observations.begin(); o_it != observations.end(); ++o_it) {
      LandmarkObs obs = {
        .id = -1,
        .x = p_it->x + o_it->x * cos(p_it->theta) - o_it->y * sin(p_it->theta),
        .y = p_it->y + o_it->x * sin(p_it->theta) + o_it->y * cos(p_it->theta),
      };
      observed.push_back(obs);
    }
    
    // Use nearest neighbors to associate each observed landmark with a map landmark ID.
    dataAssociation(predicted, observed);
    
    // Loop through each sensor observation and calculate the probability that it actually matches
    // the nearest neighbor associated above. Multiply the probabilities of each sensor reading to
    // generate the new weight for the particle.
    double total_weight = 1.0;
    for (std::vector<LandmarkObs>::const_iterator o_it = observed.begin(); o_it != observed.end(); ++o_it) {
      // Look up the map landmark by id from id assigned to the observed reading. This is a bit
      // fragile but will work as long as the 1-indexed map landmarks are added sequentially to
      // the map_landmarks list.
      Map::single_landmark_s ml = map_landmarks.landmark_list[o_it->id - 1];
      
      // Estimate the weight with the multivariate Gaussian distribution.
      double x_diffsq = (o_it->x - ml.x_f) * (o_it->x - ml.x_f);
      double y_diffsq = (o_it->y - ml.y_f) * (o_it->y - ml.y_f);
      double weight = exp(-(x_diffsq / (TWO_PI * std_x) + y_diffsq / (TWO_PI * std_y))) / (TWO_PI * std_x * std_y);
      total_weight *= weight;
    }
    p_it->weight = total_weight;
  }

  // Update weights vector with the new particle weights.
  for (int i = 0; i < num_particles; ++i) {
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  
  std::vector<Particle> sampled_particles;
  for (int i = 0; i < num_particles; ++i) {
    sampled_particles.push_back(particles[dist(gen)]);
  }
  particles.swap(sampled_particles);
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
