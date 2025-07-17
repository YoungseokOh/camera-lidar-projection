void LidarCameraCalibration::projectionPointCloud(const string& name,
    const string& imagePath, 
    const string& type, 
    const string& save_path,
    int index,
    float maxRange) 
{
Mat image;
vector<float> intrinsic = utils->getSensorInfo().intrinsicMap.at(name);
vector<float> rodriguesExtrinsic = utils->getSensorInfo().extrinsicMap.at(name);
Matrix4f extrinsic = utils->extrinsicConverter(rodriguesExtrinsic);
Matrix4f L2CExtrinsic = extrinsic * L2WMatrix;
cout << "L2CExtrinsic : " << endl << L2CExtrinsic << endl;
cout << "extrinsic : " << endl << extrinsic << endl;
image = utils->loadImage(imagePath);

Vector4f point_homo;
Vector4f extrinsic_result;
PointCloudType::Ptr data;
int u = 0, v = 0;

if (isCalib) data = cloudCluster;
else data = cloud;

for (const auto& point : *data)
{
point_homo << point.x, point.y, point.z, 1;
extrinsic_result = L2CExtrinsic * point_homo;
if (extrinsic_result(0) <= 0 || extrinsic_result(0) >=4.3 || extrinsic_result(2) >= 3) continue;
//if (extrinsic_result(0) <= -0.5 || extrinsic_result(0) >= 4.3 || extrinsic_result(2) >= 3) continue;
if (intrinsic.size() == 4 && type == "pinhole") {
float xn = -extrinsic_result(1) / extrinsic_result(0);
float yn = -extrinsic_result(2) / extrinsic_result(0);

u = intrinsic[0] * xn + intrinsic[2];
v = intrinsic[1] * yn + intrinsic[3];
}

else if (utils->getSensorInfo().camInfo.intrinsic.size() == 6 && 
utils->getSensorInfo().camInfo.cameraModel == "eucm")
{
float denom = intrinsic[0] * sqrt(extrinsic_result(0)* extrinsic_result(0) + intrinsic[1] * (extrinsic_result(1)* extrinsic_result(1) + extrinsic_result(2)* extrinsic_result(2))) + (1.f - intrinsic[0]) * extrinsic_result(0);
float xn = -extrinsic_result(1) / denom;
float yn = -extrinsic_result(2) / denom;

u = intrinsic[2] * xn + intrinsic[4];
v = intrinsic[3] * yn + intrinsic[5];
}
else if (utils->getSensorInfo().camInfo.intrinsic.size() == 18 &&
utils->getSensorInfo().camInfo.cameraModel == "vadas")
{
cv::Point2f normPt = cv::Point2f(-extrinsic_result(1), -extrinsic_result(2));
float dist = norm(normPt);
dist = dist < DBL_EPSILON ? DBL_EPSILON : dist;
float cosPhi = normPt.x / dist;
float sinPhi = normPt.y / dist;
float theta = atan2(dist, extrinsic_result(0));

float rd = 0.;
float xd = theta * intrinsic[7];
for (int i = 6; i >= 0; i--)
{
rd = rd * xd + intrinsic[i];
}
rd /= intrinsic[8];

u = (rd * cosPhi + intrinsic[9] + image.cols / 2);
v = (rd * sinPhi + intrinsic[10] + image.rows / 2);
}
if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) 
{
float norm_dist = extrinsic_result(0) / maxRange;
norm_dist = std::min(std::max(norm_dist, 0.0f), 1.0f);
cv::Scalar color = utils->calculateColorFromDistance(norm_dist);
circle(image, Point(u, v), 1, color, -1);
}
}
imshow("Projection", image);
if (!isCalib) 
{
imwrite(save_path + "projection_val" + to_string(calibIndex - 1) + to_string(index) + ".png", image);
}
else 
{
imwrite(save_path + "projection_" + to_string(index) + ".png", image);
}
waitKey(1000);
}