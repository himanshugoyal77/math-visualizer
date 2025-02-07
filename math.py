import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Optional, Tuple, Dict

class MathVisualizer:
    def plot_parabola(self, a: float, b: float, c: float, x_range: Tuple[float, float] = (-10, 10)):
        """
        Plot a parabola of the form y = ax² + bx + c
        
        Parameters:
        - a, b, c: coefficients of the quadratic equation
        - x_range: tuple of (min_x, max_x) for the plot range
        """
        x = np.linspace(x_range[0], x_range[1], 200)
        y = a * x**2 + b * x + c
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'y = {a}x² + {b}x + {c}',
            line=dict(color='blue', width=2)
        ))
        
        # Add the axis lines
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.add_vline(x=0, line_width=1, line_color="black")
        
        # Add vertex point
        vertex_x = -b / (2*a)
        vertex_y = a * vertex_x**2 + b * vertex_x + c
        fig.add_trace(go.Scatter(
            x=[vertex_x], y=[vertex_y],
            mode='markers+text',
            name='Vertex',
            text=['Vertex'],
            textposition="top center",
            marker=dict(size=10, color='red')
        ))
        
        fig.update_layout(
            title=f'Parabola: y = {a}x² + {b}x + {c}',
            xaxis_title='x',
            yaxis_title='y',
            showlegend=True
        )
        
        return fig

    def plot_ellipse(self, a: float, b: float, center: Tuple[float, float] = (0, 0)):
        """
        Plot an ellipse with semi-major axis a and semi-minor axis b
        
        Parameters:
        - a: length of semi-major axis
        - b: length of semi-minor axis
        - center: (h, k) center point of ellipse
        """
        t = np.linspace(0, 2*np.pi, 200)
        h, k = center
        x = h + a * np.cos(t)
        y = k + b * np.sin(t)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'Ellipse: (x-{h})²/{a}² + (y-{k})²/{b}² = 1',
            line=dict(color='blue', width=2)
        ))
        
        # Add the axis lines
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.add_vline(x=0, line_width=1, line_color="black")
        
        # Add foci points
        c = np.sqrt(abs(a**2 - b**2))
        if a > b:
            foci_x = [h - c, h + c]
            foci_y = [k, k]
        else:
            foci_x = [h, h]
            foci_y = [k - c, k + c]
            
        fig.add_trace(go.Scatter(
            x=foci_x, y=foci_y,
            mode='markers+text',
            name='Foci',
            text=['F₁', 'F₂'],
            textposition="top center",
            marker=dict(size=10, color='red')
        ))
        
        fig.update_layout(
            title=f'Ellipse: (x-{h})²/{a}² + (y-{k})²/{b}² = 1',
            xaxis_title='x',
            yaxis_title='y',
            showlegend=True
        )
        
        return fig

    def plot_3d_surface(self, func: callable, x_range: Tuple[float, float] = (-5, 5), 
                       y_range: Tuple[float, float] = (-5, 5), points: int = 100):
        """
        Plot a 3D surface defined by a function z = f(x, y)
        
        Parameters:
        - func: callable that takes x and y arrays and returns z values
        - x_range, y_range: tuples of (min, max) for x and y axes
        - points: number of points to use in each dimension
        """
        x = np.linspace(x_range[0], x_range[1], points)
        y = np.linspace(y_range[0], y_range[1], points)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        
        fig.update_layout(
            title='3D Surface Plot',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            )
        )
        
        return fig

    def plot_parametric_curve(self, x_func: callable, y_func: callable, 
                            t_range: Tuple[float, float], points: int = 1000):
        """
        Plot a parametric curve defined by x = x(t) and y = y(t)
        
        Parameters:
        - x_func: callable that takes t and returns x values
        - y_func: callable that takes t and returns y values
        - t_range: tuple of (min_t, max_t)
        - points: number of points to plot
        """
        t = np.linspace(t_range[0], t_range[1], points)
        x = x_func(t)
        y = y_func(t)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Parametric Curve',
            line=dict(color='blue', width=2)
        ))
        
        # Add the axis lines
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.add_vline(x=0, line_width=1, line_color="black")
        
        fig.update_layout(
            title='Parametric Curve',
            xaxis_title='x',
            yaxis_title='y',
            showlegend=True
        )
        
        return fig

    def plot_vector_field(self, u_func: callable, v_func: callable, 
                         x_range: Tuple[float, float] = (-5, 5),
                         y_range: Tuple[float, float] = (-5, 5),
                         density: int = 20):
        """
        Plot a 2D vector field defined by u(x,y) and v(x,y)
        
        Parameters:
        - u_func: callable that takes x and y arrays and returns x-component of vectors
        - v_func: callable that takes x and y arrays and returns y-component of vectors
        - x_range, y_range: tuples of (min, max) for x and y axes
        - density: number of vectors in each dimension
        """
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x, y)
        U = u_func(X, Y)
        V = v_func(X, Y)
        
        fig = go.Figure()
        
        # Add vector field using quiver plot
        for i in range(density):
            for j in range(density):
                fig.add_trace(go.Scatter(
                    x=[X[i,j], X[i,j] + U[i,j]],
                    y=[Y[i,j], Y[i,j] + V[i,j]],
                    mode='lines+markers',
                    line=dict(color='blue', width=1),
                    marker=dict(size=3),
                    showlegend=False
                ))
        
        fig.update_layout(
            title='Vector Field',
            xaxis_title='x',
            yaxis_title='y',
            showlegend=False
        )
        
        return fig

# Extend the PersonalizedTutor class with the new visualizations
class PersonalizedTutor:
    def __init__(self, api_key: str):
        # ... (previous initialization code remains the same)
        self.math_viz = MathVisualizer()
    
    def visualize_quadratic(self, a: float, b: float, c: float):
        """Visualize a quadratic function"""
        return self.math_viz.plot_parabola(a, b, c)
    
    def visualize_ellipse(self, a: float, b: float, center: Tuple[float, float] = (0, 0)):
        """Visualize an ellipse"""
        return self.math_viz.plot_ellipse(a, b, center)
    
    def visualize_3d_function(self, func: callable):
        """Visualize a 3D surface"""
        return self.math_viz.plot_3d_surface(func)
    
    def visualize_parametric(self, x_func: callable, y_func: callable, t_range: Tuple[float, float]):
        """Visualize a parametric curve"""
        return self.math_viz.plot_parametric_curve(x_func, y_func, t_range)
    
    def visualize_vector_field(self, u_func: callable, v_func: callable):
        """Visualize a vector field"""
        return self.math_viz.plot_vector_field(u_func, v_func)
    
    
# Initialize the tutor
tutor = PersonalizedTutor(api_key="your-api-key")

# Example 1: Plot a parabola
parabola_fig = tutor.visualize_quadratic(a=1, b=0, c=0)
parabola_fig.show()

# Example 2: Plot an ellipse
ellipse_fig = tutor.visualize_ellipse(a=3, b=2)
ellipse_fig.show()

# Example 3: Plot a 3D surface (e.g., paraboloid)
def paraboloid(x, y):
    return x**2 + y**2

surface_fig = tutor.visualize_3d_function(paraboloid)
surface_fig.show()

# Example 4: Plot a parametric curve (e.g., circle)
circle_x = lambda t: np.cos(t)
circle_y = lambda t: np.sin(t)
parametric_fig = tutor.visualize_parametric(circle_x, circle_y, (0, 2*np.pi))
parametric_fig.show()

# Example 5: Plot a vector field (e.g., rotational field)
def u(x, y): return -y
def v(x, y): return x
vector_field_fig = tutor.visualize_vector_field(u, v)
vector_field_fig.show()