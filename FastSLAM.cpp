#include <vector>
#include <string>  
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath> 
#include <limits>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "FastSLAM.h"

using namespace std;
using namespace Eigen;

double FastSLAM::NormalAngle(double angle)
{
	while(angle>M_PI)
	{
		angle=angle-2*M_PI;
		
		}
	while(angle<-M_PI)
	{
		angle=angle+2*M_PI;
		}
	return angle;
}

void FastSLAM::fastSLAM()
{
	string a;
	double b=0;
	double c=0;
	double d=0;

	vector<int> landmarkcountvec;
	vector<Odometry> combine;

	ifstream fin("sensor_data.dat");
	
	if(!fin)
	{
		cout<<"Error! Cannot find the sensor_data.dat file!"<<endl;
	}
	
	int landmarkcount=0;
     
    char str[200];
    int count=-1;
    while(fin.getline(str,200))
    {
		string m=str;
		istringstream iss(m);
		iss>>a;
		iss>>b;
		iss>>c;
		iss>>d;
	    
	    //cout<<a<<"\t"<<b<<"\t"<<c<<"\t"<<d<<"\t"<<endl;
	    
			if (a.compare("ODOMETRY")==0)
			{  
				if(landmarkcount>0)
					{
						landmarkcountvec.push_back(landmarkcount);
					}
				landmarkcount=0;
				Odometry Odom;
				Odom.r1 =NormalAngle(b);
				Odom.t = c;
				Odom.r2 =NormalAngle(d);
				vector<Sensor> tmpsensor;
				Odom.sensorVec=tmpsensor;
				count++;
				combine.push_back(Odom);
				
			}
			else
			{   
				landmarkcount++;
				Sensor sen;
				sen.id = b;
				sen.r = c;
				sen.theta = NormalAngle(d);
				combine[count].sensorVec.push_back(sen);
			}
	}
	
	ofstream fout("robot_pose.csv");
    if(!fout)
    {
	   cout<<"cannot output data"<<endl;
    } 
    
    int numParticles=100;
    vector<double> weightVec;
    vector<Particle> particles;
   
    int N=landmarkcountvec.size();
    //cout<<N<<endl;
    //cout<<combine.size()<<endl;
    
    double r1Noise = 0.005;
    double transNoise = 0.01;
    double r2Noise = 0.005;
    
    //Initialize the landmarks and robot poses, these are the landmark positions and robot poses at t=0
    for(int k=0; k<numParticles; k++)
    {
		//robot poses initialization
		Particle Par;
        Par.poseX=0;
        Par.poseY=0;
        Par.poseTheta=0;
        Par.weight=1.0/numParticles;
        particles.push_back(Par);
        weightVec.push_back(Par.weight);
        //cout<<particles[k].poseX<<"\t"<<particles[k].poseY<<"\t"<<particles[k].poseTheta<<"\t"<<particles[k].weight<<endl;
		
		//landmark Initialization
		for(int l=0; l<9;l++)
		{  
			particles[k].landmarks[l].observed = false;
            particles[k].landmarks[l].mu = MatrixXd::Zero(2,1);    // 2D position of the landmark
            particles[k].landmarks[l].sigma = MatrixXd::Zero(2,2); // Covariance of the landmark
  
         }
        
     }
    
    for (int t=0; t<60 ;t++) //It's OK to run just 60 times. but it appeared Segmentation fault (core dumped) if 70
    {
		std::default_random_engine generator;
		std::normal_distribution<double> distributionR1(combine[t].r1,r1Noise);
		std::normal_distribution<double> distributionTrans(combine[t].t,transNoise);
		std::normal_distribution<double> distributionR2(combine[t].r2,r2Noise);
		
		for(int k=0; k<numParticles; k++)
		{
			//Prediction. Sample a new pose from odometry information:
			
			double Dr1=distributionR1(generator);
			double Dtrans=distributionTrans(generator);
			double Dr2=distributionR2(generator);
			
			particles[k].poseX=particles[k].poseX+Dtrans*cos(particles[k].poseTheta+Dr1);
			particles[k].poseY=particles[k].poseY+Dtrans*sin(particles[k].poseTheta+Dr1);
			particles[k].poseTheta=NormalAngle(particles[k].poseTheta+Dr1+Dr2);
			//cout<<particles[k].poseX<<"\t"<<particles[k].poseY<<"\t"<<particles[k].poseTheta<<"\t"<<particles[k].weight<<endl;
			//fout<<particles[k].poseX<<" "<<particles[k].poseY<<" "<<particles[k].poseTheta<<" "<<particles[k].weight<<endl;
			
		//Measurement update. For each observed landmark j, incorporate the measurement landmark poses 
		//into the corresponding EKF, by updating the mean and covariance.
		
		//For all observed landmarks, do the following: 
		for (int l=0; l<landmarkcountvec[t] ; l++)  //
		{
            int j=combine[t].sensorVec[l].id;  //data association
             //ZObservation
			MatrixXd ZObservation(2,1);
		    ZObservation(0,0)=combine[t].sensorVec[l].r;
			ZObservation(1,0)=combine[t].sensorVec[l].theta;
	        
			//if landmark j has never seen before:
			if(particles[k].landmarks[j].observed==false)
			{
				//Initialize mean
				particles[k].landmarks[j].mu(0,0)=particles[k].poseX+combine[t].sensorVec[l].r*cos(combine[t].sensorVec[l].theta+particles[k].poseTheta);
				particles[k].landmarks[j].mu(1,0)=particles[k].poseY+combine[t].sensorVec[l].r*sin(combine[t].sensorVec[l].theta+particles[k].poseTheta);
				particles[k].landmarks[j].observed=true;
			   
			   // cout<<particles[k].landmarks[j].mu(0,0)<<"\t"<<particles[k].landmarks[j].mu(1,0)<<endl;
			    //fout<<particles[k].landmarks[j].mu(0,0)<<" "<<particles[k].landmarks[j].mu(1,0)<<endl;
			    
				//Compute expected observation according to the current estimate
				double dx=particles[k].landmarks[j].mu(0,0)-particles[k].poseX;
				double dy=particles[k].landmarks[j].mu(1,0)-particles[k].poseY;
			
				double q=dx*dx+dy*dy; 
			
				MatrixXd h(2,1);
				h<<sqrt(q),
					NormalAngle(atan2(dy,dx)-particles[k].poseTheta);
						
				// Observation Jacobian
				MatrixXd JacobianH(2,2);
				JacobianH<<dx*sqrt(q)/q, dy*sqrt(q)/q,
						-dy/q, dx/q;
					  
				//Observation Noise Qt
				MatrixXd Qt(2,2);
				Qt<<0.01,0,
					0,0.01;
				
				//Initialize covariance
				particles[k].landmarks[j].sigma=JacobianH.inverse()*Qt*(JacobianH.inverse()).transpose();
			     
			    particles[k].weight=1.0/numParticles;  //default importance weight
			    weightVec[k]=particles[k].weight;
			
			}
			else
			{

				//Compute expected observation according to the current estimate
				double dx=particles[k].landmarks[j].mu(0,0)-particles[k].poseX;
				double dy=particles[k].landmarks[j].mu(1,0)-particles[k].poseY;
			
				double q=dx*dx+dy*dy; 
			
				//Measurement Prediction
				MatrixXd h(2,1);
				h<<sqrt(q),
					NormalAngle(atan2(dy,dx)-particles[k].poseTheta);
						
				// Observation Jacobian
				MatrixXd JacobianH(2,2);
				JacobianH<<dx*sqrt(q)/q, dy*sqrt(q)/q,
						-dy/q, dx/q;
						
				//Observation Noise Qt
				MatrixXd Qt(2,2);
				Qt<<0.01,0.0,
					0,0.01;
	               
				//Compute the Kalman Gain
				MatrixXd K;
			
				MatrixXd Temp;
				Temp=JacobianH*particles[k].landmarks[j].sigma*JacobianH.transpose()+Qt;
			
				K=particles[k].landmarks[j].sigma*JacobianH.transpose()*Temp.inverse();
			
				MatrixXd Zdifference(2,1);
				Zdifference=ZObservation-h;
				Zdifference(1,0)=NormalAngle(Zdifference(1,0));
				
				//Q is the measurement covariance (pose uncertainty of the landmark estimate plus measurement noise) 
				MatrixXd Q;
				Q=JacobianH*particles[k].landmarks[j].sigma*JacobianH.transpose()+Qt;
				
				MatrixXd temp1;
				double temp2;
				temp1=Zdifference.transpose()*Q.inverse()*Zdifference;
				
			    temp2=pow((2*M_PI*Q).determinant(),-0.5)*exp(-0.5*temp1(0,0));
				particles[k].weight=particles[k].weight*temp2;
				weightVec[k]=particles[k].weight;
				
				//cout<<weightVec[k]<<endl;
				
				
			    //update mean and covariance
				particles[k].landmarks[j].mu=particles[k].landmarks[j].mu+K*Zdifference;
				particles[k].landmarks[j].sigma=(MatrixXd::Identity(2,2)-K*JacobianH)*particles[k].landmarks[j].sigma;
			    
			    
				//cout<<weightVec[k]<<endl;
			    //cout<<particles[k].poseX<<"\t"<<particles[k].poseY<<"\t"<<particles[k].poseTheta<<"\t"<<particles[k].weight<<endl;
			    //fout<<particles[k].poseX<<" "<<particles[k].poseY<<" "<<particles[k].poseTheta<<" "<<particles[k].weight<<endl;
			
			}	
			
		}	
	} 
	
	//Resampling
	for(int k=0; k<numParticles; k++)
    {
		//robot poses initialization
		Particle Par;
        Par.poseX=0;
        Par.poseY=0;
        Par.poseTheta=0;
        Par.weight=1.0/numParticles;
        newParticles.push_back(Par);
	}
	
    double weightSum=0.0;
    double weightSquareSum=0.0;
    
    
    for(int k=0; k<numParticles; k++)
    {
		//cout<<weightVec[k]<<endl;
		weightSum=weightSum+weightVec[k];
		//cout<<weightSum<<endl;
		weightSquareSum=weightSquareSum+weightVec[k]*weightVec[k];
		//cout<<weightSquareSum<<endl;
		}
		
    //Number of Effective Particles
	//Empirical measure of how well the target distribution is approximated by samples drawn from the proposal
	//neff describes "the inverse variance of the normalized particle weights"
	double neff = 1.0 / weightSquareSum;
				
    //Consider the number of effective particles, to decide whether to resample or not
    //Resampling with neff. If our approximation is close to the target, no resampling is needed.
    //We only resample when the neff drops below(<) a given threshold (0.5*numParticles);
    
    bool useNeff=false;
    if(useNeff==true)
   {
	if(neff>0.5*numParticles)   //if neff>0.5*numParticles, we don't conduct the resample. the newParticles=particles directly
	{
		particles=particles;
	}
	}
			
	//Then if neff drops below(<) 0.5*numParticles, we conduct resample
	//Initialize the step and the current position on the roulette wheel
	double step=weightSum/numParticles;
    std::random_device rd;
	std::mt19937_64 mt(rd());
	std::uniform_real_distribution<double> distribution(0,weightSum);
	double position=distribution(mt);
 
    double csum[1000];
    csum[0]=weightVec[0];
    for(int j=0; j<weightVec.size(); j++)
    {
       csum[j]=csum[j-1]+weightVec[j];
   }
 
	int idx=0;
	//Walk along the wheel to select the particles
	for(int k=0; k<numParticles; k++)
	{
		position=position+step;
		while(position>weightSum)
		{
			position=position-weightSum;    //rotate the new wheel
			idx=0;
		}
		while(position>csum[idx])
		{ 
			idx++;
		}
		particles[k]=particles[idx];
		particles[k].weight=1.0/numParticles; //After resampling, all the particles have the same weight
		cout<<particles[k].poseX<<"\t"<<particles[k].poseY<<"\t"<<particles[k].poseTheta<<"\t"<<particles[k].weight<<endl;
		fout<<particles[k].poseX<<" "<<particles[k].poseY<<" "<<particles[k].poseTheta<<" "<<particles[k].weight<<endl;
			
	} 
	  
		
	}
	return;
}
	
	
int main()
{
	FastSLAM fastslam;
	fastslam.fastSLAM();
	
	return 0;
}

