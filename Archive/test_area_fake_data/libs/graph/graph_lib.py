import graph_basic as grba
import graph_plots as grpl

import graph_setting as grset
import graph_axes as graxes
import graph_axis as graxis
import graph_subplots as grsp
class CGraph ():
    
    def __init__(self,w = 20, h = 12, lw = 2):
        self.init_variables(w = w, h = h, lw = lw)
        
    ####################### Basic functions  #######################
    figure_management = grba.figure_management
    init_variables = grba.init_variables
    savefig = grba.savefig
    init_figure = grba.init_figure
    close = grba.close
    
    update_legend = grset.update_legend
    
    ####################### Axis functions #######################
    format_xaxis = graxis.format_xaxis
    format_yaxis = graxis.format_yaxis
    
    format_axis2 = graxis.format_axis2
    color_axis = graxis.color_axis
    hide_xaxis =  graxis.hide_xaxis
    hide_yaxis =  graxis.hide_yaxis
    
    ####################### Setting functions #######################
    format_legend = grset.format_legend
    set_textRotations = grset.set_textRotations
    set_fontSizes = grset.set_fontSizes
    preprocess_data = grset.preprocess_data
    get_color = grset.get_color
    set_labels = grset.set_labels
    
    store_WidgetData = grset.store_WidgetData
    init_WidgetData = grset.init_WidgetData
    get_barwidth = grset.get_barwidth
    add_text = grset.add_text
    ######################## Axes functions #######################
    manage_axes = graxes.manage_axes
    create_axes = graxes.create_axes
    twin_axes = graxes.twin_axes
    get_axes = graxes.get_axes
    set_xlim = graxes.set_xlim
    set_ylim = graxes.set_ylim
    set_zoom = graxes.set_zoom

 
    ####################### Subplots functions #######################
    set_subplots = grsp.set_subplots
    next_subplot = grsp.next_subplot
    subplots_adjust =  grsp.subplots_adjust
    subplot2grid = grsp.subplot2grid
    apply_style = grsp.apply_style

    ####################### Basic graph functions #######################
    plot = grpl.plot
    scatter = grpl.scatter
    stem = grpl.stem
    
    bar = grpl.bar
    step = grpl.step
    
    plot_filled = grpl.plot_filled
    fill_between = grpl.fill_between
    
    add_hlines = grpl.add_hlines
    add_vlines = grpl.add_vlines

gl = CGraph()

#import numpy as np
#import matplotlib.pyplot as plt
##from matplotlib.widgets import TextBox
#fig, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.2)
#t = np.arange(-2.0, 2.0, 0.001)
#s = t ** 2
#initial_text = "t ** 2"
#l, = plt.plot(t, s, lw=2)
#
#
#def submit(text):
#    ydata = eval(text)
#    l.set_ydata(ydata)
#    ax.set_ylim(np.min(ydata), np.max(ydata))
#    plt.draw()
#
#axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
#text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
#text_box.on_submit(submit)
#
#plt.show()
