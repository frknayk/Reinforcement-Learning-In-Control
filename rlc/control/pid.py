import control as ct
import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Target/reference value

        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, feedback_value):
        # Calculate the error
        error = self.setpoint - feedback_value

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.prev_error)

        # Update the previous error
        self.prev_error = error

        # Calculate the PID output
        output = proportional + integral + derivative

        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
