import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation, cm
import matplotlib as mpl



u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
r = 10

x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))


image = plt.imread('bg.jpg') 



fig = plt.figure(facecolor='Black')
ax = plt.axes(projection='3d')

plt.axis('off')
ax.patch.set_alpha(0)

background_ax = plt.axes([0, 0, 1, 1]) # create a dummy subplot for the background
background_ax.set_zorder(-1) # set the background subplot behind the others
background_ax.imshow(image, aspect='auto') # show the backgroud image



# def init():
#     ax.plot_surface(x, y, z, rstride=2, cstride=2)

# def animate(i):
#     ax.view_init(elev=20, azim=i*4)

# ani = animation.FuncAnimation(fig, animate, init_func=init, frames=90, interval=200, blit=False)

# ani.save('file_name.gif', writer=animation.PillowWriter())
# st.image('file_name.gif')





# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))

# st.dataframe(dataframe.style.highlight_max(axis=0))

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)

# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)

R = st.slider('R', min_value=1, max_value=255)  # 👈 this is a widget
G = st.slider('G', min_value=1, max_value=255)  # 👈 this is a widget
B = st.slider('B', min_value=1, max_value=255)  # 👈 this is a widget

R = hex(R).__str__().split('x')[1]
G = hex(G).__str__().split('x')[1]
B = hex(B).__str__().split('x')[1]

if len(R) < 2:
    R = '0' + R
if len(G) < 2:
    G = '0' + G
if len(B) < 2:
    B = '0' + B

RGB = R + G + B

st.write(RGB)

ax.plot_surface(x, y, z, rstride=1, cstride=1, color='#'+RGB)
st.pyplot(fig, clear_figure=True)