import io
import base64
import datetime

import numpy as np
from fire import Fire
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Input, Output, dcc, html, no_update, Dash, State
from jupyter_dash import JupyterDash
from PIL import Image


from on2logic.utils import return_default_transforms, case_study_setup
from on2logic.model import generate_vector_for_pil_image, load_image_model


def fit_pca(n_components:int,):
    
    _, library_dataframe = case_study_setup()
    manuscript_vectors = np.vstack(library_dataframe['vector'].values)
    pca_transformed = PCA(n_components = n_components).fit_transform(manuscript_vectors).T
    for pca_dim, vectors_in_this_dim in enumerate(pca_transformed):
        library_dataframe[f'pca_{pca_dim}'] = vectors_in_this_dim
        
    return library_dataframe

def generate_pca_plot(df_to_plot, 
                      x_dim:int, 
                      y_dim:int):
    
    pca_fig = px.scatter(df_to_plot, 
                         x=f'pca_{x_dim}',
                         y=f'pca_{y_dim}')
    return pca_fig

def pca_app(mode:str, 
            n_components:int = 4,
            tooltip_alpha: float = 0.75,
            img_alpha: float = 0.7,
            width: float = 150,
            ):
    
    
    fitted_dataset = fit_pca(n_components=n_components)
    fig = generate_pca_plot(fitted_dataset)
    
    # select axes based on user choice? 
    # https://stackoverflow.com/questions/54583996/how-to-use-callback-to-update-bar-graph-in-dash
    if mode == 'inline':
        app = JupyterDash(__name__)
    else:
        app = Dash(__name__)
        
    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(
                id="graph-tooltip", background_color=f"rgba(255,255,255,{tooltip_alpha})"
            ),
        ])
    
    @app.callback(
        output=[
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
        ],
        inputs=[Input("graph-basic-2", "hoverData"),
                ],
    )
    def display_hover(hoverData, value):
        if hoverData is None:
            return False, no_update, no_update
        
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        curve_num = pt["curveNumber"]
        
        if len(fig.data) != 1:
            df_curve = curve_dict[curve_num].reset_index(drop=True)
            df_row = df_curve.iloc[num]
        else:
            df_row = df.iloc[num]
        
        img_str = 
        hoverbox_elements = [html.Img(
                        src=img_str,
                        style={
                            "width": "100%",
                            "background-color": f"rgba(255,255,255,{img_alpha})",
                        },
                    )]
        children = [
            html.Div(
                hoverbox_elements,
                style={
                    "width": f"{width}px",
                    "white-space": "normal",
                },
            )
        ]
        return True, bbox, children
    
    # define mouseover pillow image
    return app
#%%

def search_app(mode:str):
    
    # TODO load data
    _, library_dataframe = case_study_setup()
    
    image_model = load_image_model()
    transform = return_default_transforms()
    
    if mode == 'inline':
        app = JupyterDash(__name__)
    else:
        app = Dash(__name__)
        
    app.layout = html.Div([
        dcc.Upload(
            id='upload-image',
            children = html.Button('Upload File')),
        # dcc.Upload(
        # id='upload-image',
        # children=html.Div([
        #     'Drag and Drop or ',
        #     html.A('Select Files')
        # ]),
        # style={
        #     'width': '100%',
        #     'height': '60px',
        #     'lineHeight': '60px',
        #     'borderWidth': '1px',
        #     'borderStyle': 'dashed',
        #     'borderRadius': '5px',
        #     'textAlign': 'center',
        #     'margin': '10px'
        # },),
        html.Div(id='output-image-upload'),
    ])
    # @app.callback(
    #     output=[],
    #     inputs=[],
    # )
    # def display_hover():
    #     return
    
   # TODO convert uploaded image to PIL image, calculate vector
    
    # TODO calculate cosine similarity between uploaded image to library
    def parse_contents(contents, filename, date):
        img_content = html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(src=contents),
            html.Hr(),
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })
            ])
        return img_content
    
    @app.callback(Output('output-image-upload', 'children'),
                Input('upload-image', 'contents'),
                State('upload-image', 'filename'),
                State('upload-image', 'last_modified'))
    def update_output(contents, filename, last_modified):
        if contents is not None:
            children = parse_contents(contents, filename, last_modified)
            pil_image = pil_image_from_html_img_src(contents)
            vector = generate_vector_for_pil_image(pil_image=pil_image, image_mdoel=image_model, torchvision_transform=transform)
            return children
    return app

def pil_image_from_html_img_src(img_src):
    img_string = img_src.split(',')[1]
    im_bytes = base64.b64decode(img_string)   # im_bytes is a binary image
    im_file = io.BytesIO(im_bytes)
    pil_image = Image.open(im_file)
    return pil_image

def run_search_app(mode:str = 'inline', port: int = 8701, height:int = 1000):
    dash_app = search_app(mode)

    dash_app.run_server(mode=mode, port=port, height=height)
    return
run_search_app()

#%%
if __name__ == '__main__':
    Fire(run_search_app)