#ifndef FASTSLAM_H
#define FASTSLAM_H

#include <vector>
#include <string>  
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath> 
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


class FastSLAM{
	public:
	
	struct Sensor
	{
		int id;
		double r;
		double theta;
	};
	
	struct landmark
	{
		bool observed;
		MatrixXd mu;
		MatrixXd sigma;
		
     };
     
	struct Odometry
	{
		double r1;
		double t;
		double r2;
		vector<Sensor> sensorVec;

	};
	
	struct Particle
	{
		double weight;
		double poseX;
		double poseY;
		double poseTheta;
		landmark landmarks[9];
      };
	
     vector<Particle> particles;
     vector<Particle> newParticles;
     
     double NormalAngle(double angle);
     
     void fastSLAM();
     
	
};


#endif // FASTSLAM_H
