Matrix4f Utils::extrinsicConverter(const vector<float>& rodrigues) 
{
	// input order (tx, ty, tz, rx, ry, rz)
	float th = sqrt((rodrigues[3] * rodrigues[3] + rodrigues[4] * rodrigues[4] + rodrigues[5] * rodrigues[5]));
	float thInv = 1.f / th;
	Matrix4f T = Matrix4f::Identity();

	float u1 = rodrigues[3] * thInv;
	float u2 = rodrigues[4] * thInv;
	float u3 = rodrigues[5] * thInv;
	float sinth = sin(th);
	float costhVar = 1.f - cos(th);

	T(0, 0) = 1.0f + costhVar * (u1 * u1 - 1.0f);
	T(2, 2) = 1.0f + costhVar * (u3 * u3 - 1.0f);
	T(1, 1) = 1.0f + costhVar * (u2 * u2 - 1.0f);

	T(0, 1) = -sinth * u3 + costhVar * u1 * u2;
	T(0, 2) = sinth * u2 + costhVar * u1 * u3;
	T(1, 2) = -sinth * u1 + costhVar * u2 * u3;

	T(1, 0) = sinth * u3 + costhVar * u2 * u1;
	T(2, 0) = -sinth * u2 + costhVar * u3 * u1;
	T(2, 1) = sinth * u1 + costhVar * u3 * u2;

	T(0, 3) = rodrigues[0];
	T(1, 3) = rodrigues[1];
	T(2, 3) = rodrigues[2];

	return T;
}