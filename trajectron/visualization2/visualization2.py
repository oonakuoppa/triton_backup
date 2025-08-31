from utils import prediction_output_to_trajectories
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        # History line
        ax.plot(history[:, 0], history[:, 1], 'k--', label='History')

        # Predictions
        for sample_num in range(predictions.shape[1]):
            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color='red', linestyle='--', linewidth=1.0, alpha=line_alpha, label='Future prediction')

        # Present position as circle
        circle = plt.Circle((history[-1, 0], history[-1, 1]),
                            0.5, facecolor='orange', edgecolor='k', lw=1.0, zorder=4)
        ax.add_artist(circle)

    ax.axis('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Prediction")

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', label='History'),
        Line2D([0], [0], color='red', linestyle='--', label='Future prediction'),
        Line2D([0], [0], marker='o', color='orange', label='Present', markersize=10, linestyle='')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, **kwargs)
