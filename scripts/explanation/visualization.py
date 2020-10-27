import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

poly_dict = np.load('explanation/poly_dict.npy', allow_pickle=True)


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = poly_dict.item()['A']
    colors = [color, 'white']
    for i, polygon_coords in enumerate(a_polygon_coords):
        ppoly = (np.array([1, height])[None,:]*polygon_coords +
                 np.array([left_edge, base])[None,:])
        ax.add_patch(matplotlib.patches.Polygon(ppoly, facecolor=colors[i], edgecolor=color))
        base_poly = (np.array([1, 0.0])[None,:]*polygon_coords +
                     np.array([left_edge, 0.0])[None,:])
        ax.add_patch(matplotlib.patches.Polygon(base_poly, facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    a_polygon_coords = poly_dict.item()['C']
    for polygon_coords in a_polygon_coords:
        ppoly = (np.array([1, height])[None, :] * polygon_coords +
                 np.array([left_edge, base])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(ppoly, facecolor=color, edgecolor=color))
        base_poly = (np.array([1, 0.0])[None, :] * polygon_coords +
                     np.array([left_edge, 0.0])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(base_poly, facecolor=color, edgecolor=color))


def plot_g(ax, base, left_edge, height, color):
    a_polygon_coords = poly_dict.item()['G']
    for polygon_coords in a_polygon_coords:
        ppoly = (np.array([1, height])[None, :] * polygon_coords +
                 np.array([left_edge, base])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(ppoly, facecolor=color, edgecolor=color))
        base_poly = (np.array([1, 0.0])[None, :] * polygon_coords +
                     np.array([left_edge, 0.0])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(base_poly, facecolor=color, edgecolor=color))


def plot_t(ax, base, left_edge, height, color):
    a_polygon_coords = poly_dict.item()['T']
    for polygon_coords in a_polygon_coords:
        ppoly = (np.array([1, height])[None, :] * polygon_coords +
                 np.array([left_edge, base])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(ppoly, facecolor=color, edgecolor=color))
        base_poly = (np.array([1, 0.0])[None, :] * polygon_coords +
                     np.array([left_edge, 0.0])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(base_poly, facecolor=color, edgecolor=color))


def plot_u(ax, base, left_edge, height, color):
    a_polygon_coords = poly_dict.item()['U']
    for polygon_coords in a_polygon_coords:
        ppoly = (np.array([1, height])[None, :] * polygon_coords +
                 np.array([left_edge, base])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(ppoly, facecolor=color, edgecolor=color))
        base_poly = (np.array([1, 0.0])[None, :] * polygon_coords +
                     np.array([left_edge, 0.0])[None, :])
        ax.add_patch(matplotlib.patches.Polygon(base_poly, facecolor=color, edgecolor=color))


default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_u}


def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 4 and array.shape[1] != 4):
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
                # plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos]) #- 0.1
            max_height = np.max(heights_at_positions[start_pos:end_pos]) # + 0.1
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False,
                                             linestyle='dashed'))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)


def plot_weights(array,
                 figsize=(20, 2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    plot_weights_given_ax(ax=ax, array=array,
                           height_padding_factor=height_padding_factor,
                           length_padding=length_padding,
                           subticks_frequency=subticks_frequency,
                           colors=colors,
                           plot_funcs=plot_funcs,
                           highlight=highlight)


