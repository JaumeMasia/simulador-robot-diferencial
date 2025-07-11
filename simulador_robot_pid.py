import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def simular_pid(x_obj, y_obj, kp_dist, kp_theta):
    dt = 0.05
    T = 20
    v_max = 0.3
    x, y, theta = 0.0, 0.0, 0.0
    x_list, y_list = [x], [y]

    pid_theta = PID(Kp=kp_theta, Ki=0.0, Kd=0.1)
    pid_dist = PID(Kp=kp_dist, Ki=0.0, Kd=0.05)

    pid_theta.reset()
    pid_dist.reset()

    for _ in np.arange(0, T, 0.05):
        dx = x_obj - x
        dy = y_obj - y
        distancia = math.hypot(dx, dy)
        angulo_deseado = math.atan2(dy, dx)
        error_theta = math.atan2(math.sin(angulo_deseado - theta), math.cos(angulo_deseado - theta))

        w = pid_theta.update(error_theta, dt)
        v = pid_dist.update(distancia, dt)
        v = max(min(v, v_max), -v_max)

        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt

        x_list.append(x)
        y_list.append(y)

        if distancia < 0.05:
            break

    return x_list, y_list, theta

st.title("Simulador Robot Diferencial con PID")
col1, col2 = st.columns(2)
x_obj = col1.slider("X objetivo", -5.0, 5.0, 2.0, step=0.1)
y_obj = col2.slider("Y objetivo", -5.0, 5.0, 2.0, step=0.1)

with st.expander("Ajustes avanzados"):
    kp_dist = st.slider("Kp Distancia", 0.1, 3.0, 1.0, step=0.1)
    kp_theta = st.slider("Kp Ángulo", 0.1, 5.0, 2.0, step=0.1)

if st.button("Ejecutar Simulación"):
    x_list, y_list, theta = simular_pid(x_obj, y_obj, kp_dist, kp_theta)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x_list, y_list, 'b-', label="Trayectoria")
    ax.plot(x_obj, y_obj, 'go', label="Objetivo")
    ax.quiver(x_list[-1], y_list[-1],
              math.cos(theta), math.sin(theta),
              angles='xy', scale_units='xy', scale=0.5, color='r',
              label='Orientación')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
