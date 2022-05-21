
from fire import Fire
from dash import Input, Output, dcc, html, no_update, Dash
from jupyter_dash import JupyterDash

def pca_app(mode:str, n_dim:int ):
    
    if mode == 'inline':
        app = JupyterDash(__name__)
    else:
        app = Dash(__name__)
        
    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
    ])
    
    @app.callback(
        output=[],
        inputs=[],
    )
    def display_hover():
        return
    # define mouseover pillow image
    return app

def search_app(mode:str):
    
    if mode == 'inline':
        app = JupyterDash(__name__)
    else:
        app = Dash(__name__)
        
    app.layout = html.Div([])
    @app.callback(
        output=[],
        inputs=[],
    )
    def display_hover():
        return
    
    return app

def run_search_app(mode:str = 'inline', port: int = 8701, height:int = 1000):
    dash_app = search_app(mode)

    dash_app.run_server(mode=mode, port=port, height=height)
    return
if __name__ == '__main__':
    Fire(run_search_app)