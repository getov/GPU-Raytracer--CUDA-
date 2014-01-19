#include "OrenNayar.cuh"

__device__
OrenNayar::OrenNayar(const Color& diffuseColor, const float& roughness)
	: Shader(diffuseColor)
	, sigma(roughness)
{
}

__device__
Color OrenNayar::shade(Ray ray, const IntersectionData& data)
{
	Vector N = faceforward(ray.dir, data.normal);

	Color diffuseColor = this->_color;
	Color lightContrib = ambientLight;

	Color incomingLight = lightColor * lightPower / (data.p - lightPos).lengthSqr();

	double A = 1.0 - 0.5 * (sigma * sigma / (sigma * sigma + 0.33));
	double B = 0.45 * (sigma * sigma / (sigma * sigma + 0.009));

	Vector cameraDir = cameraPos - data.p;
	cameraDir.normalize();
	Vector lightDir = lightPos - data.p;
	lightDir.normalize();

	// cosine of the angle between the surface normal
	// and lighDir and cameraDir vectors respectively
	double cosThetaLight  = dot(lightDir, N);
	double cosThetaCamera = dot(cameraDir, N);

	// angle between the surface normal and cameraDir vector
	double thetaCamera = std::acos(cosThetaCamera); 
	// angle between the surface normal and lightDir vector
	double thetaLight  = std::acos(cosThetaLight);

	double alpha = thetaLight > thetaCamera ? thetaLight : thetaCamera;
	double beta  = thetaLight < thetaCamera ? thetaLight : thetaCamera;

	Vector cameraDirProjection = cameraDir - N * cosThetaCamera;
	cameraDirProjection.normalize();
	Vector lightDirProjection = lightDir - N * cosThetaCamera;
	lightDirProjection.normalize();

	// cosine of the difference between the azimuthial angles phi_light and phi_camera
	double azimuthDiffCos = dot(cameraDirProjection, lightDirProjection);

	azimuthDiffCos = azimuthDiffCos > 0.0 ? azimuthDiffCos : 0.0;
	Color reflectedLight = cosThetaLight * 
						(A + B * azimuthDiffCos * std::sin(alpha) * std::tan(beta)) *
		                  incomingLight;

	if (testVisibility(data.p + N * 1e-3, lightPos)) 
	{
		lightContrib += reflectedLight;
	}

	return diffuseColor * lightContrib;
}