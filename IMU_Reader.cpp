#include "Arduino_BMI270_BMM150.h"

void setup() {
  Serial.begin(9600);
  while (!Serial); // Wait for the serial port to connect - necessary for Leonardo/Micro
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1); // Infinite loop to halt operation
  }
}

void loop() {
  float mx, my, mz; // Magnetometer values
  float gx, gy, gz; // Gyroscope values
  float ax, ay, az; // Accelerometer values

  bool mAvailable = IMU.magneticFieldAvailable();
  bool gAvailable = IMU.gyroscopeAvailable();
  bool aAvailable = IMU.accelerationAvailable();

  if (mAvailable) {
    IMU.readMagneticField(mx, my, mz);
  }

  if (gAvailable) {
    IMU.readGyroscope(gx, gy, gz);
  }

  if (aAvailable) {
    IMU.readAcceleration(ax, ay, az);
  }

  // Now print all available data in one line
  if (mAvailable || gAvailable || aAvailable) {
    Serial.print("Mag x:");
    Serial.print(mx);
    Serial.print(" y:");
    Serial.print(my);
    Serial.print(" z:");
    Serial.print(mz);
    Serial.print(" | Gyro x:");
    Serial.print(gx);
    Serial.print(" y:");
    Serial.print(gy);
    Serial.print(" z:");
    Serial.print(gz);
    Serial.print(" | Acc x:");
    Serial.print(ax);
    Serial.print(" y:");
    Serial.print(ay);
    Serial.print(" z:");
    Serial.println(az);
  }

  // Add a delay to limit data rate
  delay(100);
}
