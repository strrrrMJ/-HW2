import matplotlib.pyplot as plt
import numpy as np
import h5py

from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 定义存储数据的文件名
file_name = 'indy_20160407_02.mat'
# 使用h5py库读取.mat格式数据
with h5py.File(file_name, 'r') as file:
    chan_names = file['chan_names'][:]
    cursor_pos = file['cursor_pos'][:]
    finger_pos = file['finger_pos'][:]
    spikes = file['spikes'][:]
    t = file['t'][:]
    target_pos = file['target_pos'][:]
    wf = file['wf'][:]

    all_sorted_spike_times = []
    for channel_index in range(file['spikes'].shape[1]):
        channel_sorted_units = []
        for unit_index in range(1, file['spikes'].shape[0]):
            ref = file['spikes'][unit_index, channel_index]
            if ref:
                spikes = file[ref][:].flatten()
                if spikes.size > 0:
                    channel_sorted_units.append(spikes)
        all_sorted_spike_times.append(channel_sorted_units)

    all_spikes = []

    for i in range(spikes.shape[0]):
        unit_spikes = []
        for j in range(spikes.shape[1]):
            ref = spikes[i, j]
            if ref:
                spikes_data = file[ref][:]
                unit_spikes.append(spikes_data)
        all_spikes.append(unit_spikes)


# TASK1
target_x = target_pos[0, :]
target_y = target_pos[1, :]
angles = np.degrees(np.arctan2(target_y, target_x))
angles = np.mod(angles, 360)
num_dir = 8
dir_bins = np.arange(0, 360+360 / num_dir, 360 / num_dir)
hist, _ = np.histogram(angles, bins=dir_bins)
bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2

s = UnivariateSpline(bin_centers, hist, s=0.9)
x_smth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
y_smth = s(x_smth)

plt.plot(x_smth, y_smth, label='Tuning Curve')
plt.title('Tuning Curve of Movement Directions')
plt.xlabel('Direction (degrees)')
plt.ylabel('Frequency')
plt.xticks(dir_bins)
plt.legend()
plt.show()

t = np.ravel(t)
delta_t = np.diff(t)
delta_pos = np.diff(cursor_pos, axis=1)
delta_t[delta_t == 0] = np.finfo(float).eps
v = np.sqrt((delta_pos[0] / delta_t) ** 2 + (delta_pos[1] / delta_t) ** 2)

fin_pos_mm = finger_pos * 10
fin_pos_xy = fin_pos_mm[:2, :]

delta_fin_pos = np.diff(fin_pos_xy, axis=1)
delta_t[delta_t == 0] = np.finfo(float).eps
fin_v = np.sqrt((delta_fin_pos[0] / delta_t) ** 2 + (delta_fin_pos[1] / delta_t) ** 2)

plt.hist(fin_v, bins=50, alpha=0.75)
plt.title('Finger Velocity Distribution')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Frequency')
plt.show()

# TASK2
sorted_units_spikes = all_spikes[1:]
total_time = t[-1] - t[0]
dischrg_rates = []
for unit in sorted_units_spikes:
    unit_rates = []
    for channel_spikes in unit:
        rate = len(channel_spikes) / total_time
        unit_rates.append(rate)
    dischrg_rates.append(unit_rates)


bins_cnt = 20
v_bins, bin_edges = np.histogram(fin_v, bins=bins_cnt)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

sorted_spike_times_flat = np.concatenate([
    spikes.flatten() for unit in sorted_units_spikes[1:] for channel in unit for spikes in channel if spikes.size > 0
])
sorted_spike_indices = np.searchsorted(t[1:], sorted_spike_times_flat)
sorted_spike_indices = sorted_spike_indices[sorted_spike_indices < len(fin_v)]

sorted_spike_bins = np.digitize(fin_v[sorted_spike_indices], bin_edges) - 1
sorted_spikes_per_bin = np.zeros(bins_cnt)
sorted_time_per_bin = np.zeros(bins_cnt)

for idx, bin_idx in enumerate(sorted_spike_bins):
    if 0 <= bin_idx < bins_cnt:
        sorted_spikes_per_bin[bin_idx] += 1
        sorted_time_per_bin[bin_idx] += delta_t[sorted_spike_indices[idx] - 1]

sorted_time_per_bin[sorted_time_per_bin == 0] = np.finfo(float).eps
sorted_average_rates_per_bin = sorted_spikes_per_bin / sorted_time_per_bin

plt.plot(bin_centers, sorted_average_rates_per_bin, marker='o')
plt.title('Neuronal Tuning Curve Based on Finger Velocity (Sorted Units Only)')
plt.xlabel('Finger Velocity (mm/s)')
plt.ylabel('Average Discharge Rate (spikes/s)')
plt.show()


y = sorted_average_rates_per_bin 
X = bin_centers.reshape(-1, 1)  

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)  
r_squared = r2_score(y, y_pred)     
print("r_squared = ", r_squared)   

plt.scatter(X, y, color='black', label='Actual Discharge Rates')
plt.plot(X, y_pred, color='blue', linewidth=2, label='Predicted Discharge Rates')
plt.title('Neuronal Discharge Rates vs. Velocity')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Discharge Rate (spikes/s)')
plt.legend()
plt.show()


pre_stimulus_time = 1.0
post_stimulus_time = 1.0
bin_width = 0.01  #
bins = np.arange(-pre_stimulus_time, post_stimulus_time, bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2

adjusted_spike_times = sorted_spike_times_flat - t[0]
psth_counts, _ = np.histogram(adjusted_spike_times, bins=bins)
psth_rates = psth_counts / (bin_width * len(sorted_units_spikes[1:]))  

plt.bar(bin_centers, psth_rates, width=bin_width, color='grey')
plt.title('PSTH of Neuronal Discharge Rates')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

acceleration = np.diff(fin_v) / np.mean(delta_t)

adjusted_spike_indices = sorted_spike_indices[sorted_spike_indices < len(acceleration) + 1]

acceleration_bins = np.linspace(np.min(acceleration), np.max(acceleration), bins_cnt + 1)
acceleration_bin_centers = (acceleration_bins[:-1] + acceleration_bins[1:]) / 2

spikes_per_acceleration_bin = np.zeros(bins_cnt)
time_in_acceleration_bins = np.zeros(bins_cnt)

for spike_index in adjusted_spike_indices:
    acceleration_value = acceleration[spike_index - 1]  #
    bin_index = np.digitize(acceleration_value, acceleration_bins) - 1
    if 0 <= bin_index < bins_cnt:
        spikes_per_acceleration_bin[bin_index] += 1
        time_in_acceleration_bins[bin_index] += delta_t[spike_index - 1]

time_in_acceleration_bins[time_in_acceleration_bins == 0] = np.finfo(float).eps
average_rates_per_acceleration_bin = spikes_per_acceleration_bin / time_in_acceleration_bins

plt.plot(acceleration_bin_centers, average_rates_per_acceleration_bin, marker='o')
plt.title('Neuronal Tuning Curve Based on Acceleration')
plt.xlabel('Acceleration (mm/s²)')
plt.ylabel('Average Discharge Rate (spikes/s)')
plt.show()


# TASK3
A = np.eye(2)
H = np.eye(2)
Q = np.eye(2) * 0.001
R = np.eye(2) * 0.001
x_estimate = cursor_pos[:, 0]
P_estimate = np.eye(2) * 1
estimated_positions = np.zeros_like(cursor_pos)

for i in range(cursor_pos.shape[1]):
    x_predict = A @ x_estimate
    P_predict = A @ P_estimate @ A.T + Q
    K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
    x_estimate = x_predict + K @ (cursor_pos[:, i] - H @ x_predict)
    P_estimate = (np.eye(2) - K @ H) @ P_predict

    estimated_positions[:, i] = x_estimate


plt.figure(figsize=(14, 7))
plt.plot(estimated_positions[0, :1000], estimated_positions[1, :1000], label='Estimated Position', color='red', linestyle='--', linewidth=1)
plt.title('Estimated Cursor Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show() 


plt.figure(figsize=(14, 7))
plt.plot(estimated_positions[0, :], estimated_positions[1, :], label='Estimated Position', color='red', linestyle='--', linewidth=1)
plt.title('Estimated Cursor Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show() 

def kalman_filter(Q_val, R_val, P_val):
    Q = np.eye(2) * Q_val
    R = np.eye(2) * R_val
    P_estimate = np.eye(2) * P_val
    x_estimate = cursor_pos[:, 0]
    estimated_positions = np.zeros_like(cursor_pos)

    for i in range(cursor_pos.shape[1]):
        x_predict = A @ x_estimate
        P_predict = A @ P_estimate @ A.T + Q
        K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
        x_estimate = x_predict + K @ (cursor_pos[:, i] - H @ x_predict)
        P_estimate = (np.eye(2) - K @ H) @ P_predict
        estimated_positions[:, i] = x_estimate
    return estimated_positions


parameters = [(0.001, 0.001, 1), (0.1, 0.1, 10),(1000, 1000, 0.001), (1, 1, 10), (0.001, 0.001, 1000),]

estimations = {}
for i, (Q_val, R_val, P_val) in enumerate(parameters):
    estimations[f'Q={Q_val}, R={R_val}, P={P_val}'] = kalman_filter(Q_val, R_val, P_val)

data_subset_length = 5000

def kalman_filter_subset(Q_val, R_val, P_val, data_length):
    Q = np.eye(2) * Q_val
    R = np.eye(2) * R_val
    P_estimate = np.eye(2) * P_val
    x_estimate = cursor_pos[:, 0]
    estimated_positions = np.zeros((2, data_length))

    for i in range(data_length):
        x_predict = A @ x_estimate
        P_predict = A @ P_estimate @ A.T + Q
        K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
        x_estimate = x_predict + K @ (cursor_pos[:, i] - H @ x_predict)
        P_estimate = (np.eye(2) - K @ H) @ P_predict
        estimated_positions[:, i] = x_estimate
    return estimated_positions

estimations_subset = {}
for i, (Q_val, R_val, P_val) in enumerate(parameters):
    estimations_subset[f'Q={Q_val}, R={R_val}, P={P_val}'] = kalman_filter_subset(Q_val, R_val, P_val, data_subset_length)

colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
plt.figure(figsize=(14, 14))

for i, (desc, estimated_positions_subset) in enumerate(estimations_subset.items()):
    plt.subplot(len(parameters), 1, i+1)
    plt.plot(estimated_positions_subset[0, :], estimated_positions_subset[1, :], label=f'Estimated Position ({desc})', linestyle='--', color=colors[i], linewidth=1)
    plt.title(f'Parameter Set: {desc}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

delta_t = np.diff(t.flatten())
mean_delta_t = np.mean(delta_t)
A_extended = np.array([[1, 0, mean_delta_t, 0],
                       [0, 1, 0, mean_delta_t],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

H_extended = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])

Q_extended = np.eye(4) * 0.001
R_extended = np.eye(2) * 0.001
x_estimate_extended = np.array([cursor_pos[0, 0], cursor_pos[1, 0], 0, 0])
P_estimate_extended = np.eye(4)
P_estimate_extended[2:, 2:] *= 10
estimated_positions_extended = np.zeros((2, data_subset_length))
estimated_velocities_extended = np.zeros((2, data_subset_length))

for i in range(data_subset_length):
    x_predict_extended = A_extended @ x_estimate_extended
    P_predict_extended = A_extended @ P_estimate_extended @ A_extended.T + Q_extended
    K_extended = P_predict_extended @ H_extended.T @ np.linalg.inv(H_extended @ P_predict_extended @ H_extended.T + R_extended)
    x_estimate_extended = x_predict_extended + K_extended @ (cursor_pos[:, i] - H_extended @ x_predict_extended)
    P_estimate_extended = (np.eye(4) - K_extended @ H_extended) @ P_predict_extended

    estimated_positions_extended[:, i] = x_estimate_extended[:2]
    estimated_velocities_extended[:, i] = x_estimate_extended[2:]

plt.figure(figsize=(14, 7))
plt.plot(estimated_positions_extended[0, :], estimated_positions_extended[1, :], label='Estimated Position (with velocity)', linestyle='--', color='orange', linewidth=1)

plt.title('Extended Kalman Filter with Position and Velocity')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



mse_position_only = mean_squared_error(cursor_pos[:, :data_subset_length].T, estimated_positions[:, :data_subset_length].T)
print("MSE for the position-only Kalman filter: ",mse_position_only)
mse_position_velocity = mean_squared_error(cursor_pos[:, :data_subset_length].T, estimated_positions_extended.T)

mse_values = [mse_position_only, mse_position_velocity]
labels = ['Position Only', 'Position + Velocity']

plt.figure(figsize=(10, 6))
plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.title('Comparison of MSE - Position Only vs Position + Velocity')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()


A_extended_acc = np.array([[1, 0, mean_delta_t, 0, 0.5 * mean_delta_t ** 2, 0],[0, 1, 0, mean_delta_t, 0, 0.5 * mean_delta_t ** 2],[0, 0, 1, 0, mean_delta_t, 0],[0, 0, 0, 1, 0, mean_delta_t],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])


H_extended_acc = np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0]])
Q_extended_acc = np.eye(6) * 0.001
Q_extended_acc[4:, 4:] *= 10
R_extended_acc = np.eye(2) * 0.001
x_estimate_extended_acc = np.array([cursor_pos[0, 0], cursor_pos[1, 0], 0, 0, 0, 0])
P_estimate_extended_acc = np.eye(6)
P_estimate_extended_acc[2:4, 2:4] *= 10
P_estimate_extended_acc[4:, 4:] *= 100

estimated_positions_extended_acc = np.zeros((2, data_subset_length))

for i in range(data_subset_length):
    x_predict_extended_acc = A_extended_acc @ x_estimate_extended_acc
    P_predict_extended_acc = A_extended_acc @ P_estimate_extended_acc @ A_extended_acc.T + Q_extended_acc

    K_extended_acc = P_predict_extended_acc @ H_extended_acc.T @ np.linalg.inv( H_extended_acc @ P_predict_extended_acc @ H_extended_acc.T + R_extended_acc)
    x_estimate_extended_acc = x_predict_extended_acc + K_extended_acc @ ( cursor_pos[:, i] - H_extended_acc @ x_predict_extended_acc)
    P_estimate_extended_acc = (np.eye(6) - K_extended_acc @ H_extended_acc) @ P_predict_extended_acc

    estimated_positions_extended_acc[:, i] = x_estimate_extended_acc[:2]

mse_position_velocity_acc = mean_squared_error(cursor_pos[:, :data_subset_length].T, estimated_positions_extended_acc.T)
plt.figure(figsize=(14, 7))

plt.plot(estimated_positions_extended[0, :], estimated_positions_extended[1, :], label='Estimated Position (with velocity)', linestyle='--', color='orange', linewidth=1)
plt.plot(estimated_positions_extended_acc[0, :], estimated_positions_extended_acc[1, :], label='Estimated Position (with velocity and acceleration)', linestyle='--', color='green', linewidth=1)

plt.title('Extended Kalman Filter with Position, Velocity, and Acceleration')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# MSE
mse_values = [mse_position_only, mse_position_velocity, mse_position_velocity_acc]
labels = ['Position Only', 'Position + Velocity', 'Position + Velocity + Acceleration']

plt.figure(figsize=(10, 6))
plt.bar(labels, mse_values, color=['blue', 'orange', 'green'])
plt.title('Comparison of MSE for Different Kalman Filter Models')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()



def create_features_labels(positions, num_past_points=5):
    features = []
    labels = []
    for i in range(num_past_points, len(positions[0])):
        feature = positions[:, i - num_past_points:i].flatten()
        label = positions[:, i]
        features.append(feature)
        labels.append(label)

    return np.array(features), np.array(labels)


features, labels = create_features_labels(cursor_pos[:, :data_subset_length], num_past_points=5)
split_index = int(features.shape[0] * 0.8)
features_train, features_test = features[:split_index], features[split_index:]
labels_train, labels_test = labels[:split_index], labels[split_index:]

linear_model = LinearRegression()
linear_model.fit(features_train, labels_train)
predictions_linear = linear_model.predict(features_test)
mse_linear_regression = mean_squared_error(labels_test, predictions_linear)
print("MSE for the position-only Linear Regression: ",mse_linear_regression)



mse_linear_kalman = [mse_position_only, mse_linear_regression]
labels = ['Kalman filter', 'Linear Regression']

plt.figure(figsize=(10, 6))
plt.bar(labels, mse_linear_kalman, color=['blue', 'orange'])
plt.title('Comparison of MSE - Kalman filter vs Linear Regression')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()