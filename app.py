import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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

background_ax = plt.axes([0, 0, 1, 1])
background_ax.set_zorder(-1)
background_ax.imshow(image, aspect='auto')


st.markdown(
    """# :star: Celestial Bodies Classifier :comet:"""
    )

container = st.container(border=True)

container.markdown("""### Build yours""")

space, configs = container.columns(2)

with configs:
    R = st.slider('R', min_value=1, max_value=255)
    G = st.slider('G', min_value=1, max_value=255)
    B = st.slider('B', min_value=1, max_value=255)

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

with space:
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='#'+RGB)
    st.pyplot(fig, clear_figure=True)

st.button('Classify', type='primary')