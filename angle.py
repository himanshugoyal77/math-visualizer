import plotly.graph_objects as go
import numpy as np

class AngleVisualizer:
    def create_angle_plot(self, initial_angle=45):
        """Create an interactive angle visualization with a slider"""
        
        # Initial angle in radians
        angle_rad = np.deg2rad(initial_angle)
        
        # Create figure
        fig = go.Figure()
        
        # Add fixed ray (horizontal line)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 0],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Fixed Ray'
        ))
        
        # Add movable ray
        fig.add_trace(go.Scatter(
            x=[0, np.cos(angle_rad)],
            y=[0, np.sin(angle_rad)],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Movable Ray'
        ))
        
        # Add arc
        theta = np.linspace(0, angle_rad, 50)
        r = 0.3  # radius of arc
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta),
            y=r*np.sin(theta),
            mode='lines',
            line=dict(color='blue', width=2),
            name='Arc'
        ))
        
        # Add angle label
        fig.add_trace(go.Scatter(
            x=[r/2*np.cos(angle_rad/2)],
            y=[r/2*np.sin(angle_rad/2)],
            mode='text',
            text=[f'{initial_angle}°'],
            textposition='middle center',
            name='Angle'
        ))

        # Add slider
        fig.update_layout(
            sliders=[{
                'currentvalue': {'prefix': 'Angle: ', 'suffix': '°'},
                'pad': {'t': 50},
                'steps': [
                    {
                        'method': 'restyle',
                        'args': [
                            {
                                'x': [
                                    [0, 1],  # fixed ray
                                    [0, np.cos(np.deg2rad(angle))],  # movable ray
                                    [r*np.cos(t) for t in np.linspace(0, np.deg2rad(angle), 50)],  # arc
                                    [r/2*np.cos(np.deg2rad(angle)/2)]  # label
                                ],
                                'y': [
                                    [0, 0],  # fixed ray
                                    [0, np.sin(np.deg2rad(angle))],  # movable ray
                                    [r*np.sin(t) for t in np.linspace(0, np.deg2rad(angle), 50)],  # arc
                                    [r/2*np.sin(np.deg2rad(angle)/2)]  # label
                                ],
                                'text': [[f'{angle}°']]  # update angle text
                            }
                        ],
                        'label': str(angle)
                    } for angle in range(0, 361, 5)
                ]
            }]
        )
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                range=[-1.2, 1.2],
                zeroline=True,
                showgrid=True
            ),
            yaxis=dict(
                range=[-1.2, 1.2],
                zeroline=True,
                showgrid=True,
                scaleanchor='x',
                scaleratio=1
            ),
            title='Interactive Angle'
        )
        
        return fig

# Add to PersonalizedTutor class
def visualize_angle(self, initial_angle=45):
    """Create an interactive angle visualization"""
    viz = AngleVisualizer()
    return viz.create_angle_plot(initial_angle)

fig = visualize_angle(45)
fig.show()