import io
import base64
import logging
import datetime

import numpy as np
from fire import Fire
# from sklearn.decomposition import PCA
import plotly.express as px
from dash import Input, Output, dcc, html, no_update, Dash, State
from jupyter_dash import JupyterDash
from PIL import Image


from on2logic.utils import return_default_transforms, case_study_setup, load_case_study_dataframe
from on2logic.model import generate_vector_for_pil_image, load_image_model
from on2logic.plot import perform_top_n_search

# def fit_pca(n_components:int,):
    
#     _, library_dataframe = case_study_setup()
#     manuscript_vectors = np.vstack(library_dataframe['vector'].values)
#     pca_transformed = PCA(n_components = n_components).fit_transform(manuscript_vectors).T
#     for pca_dim, vectors_in_this_dim in enumerate(pca_transformed):
#         library_dataframe[f'pca_{pca_dim}'] = vectors_in_this_dim
        
#     return library_dataframe

# def generate_pca_plot(df_to_plot, 
#                       x_dim:int, 
#                       y_dim:int):
    
#     pca_fig = px.scatter(df_to_plot, 
#                          x=f'pca_{x_dim}',
#                          y=f'pca_{y_dim}')
#     return pca_fig

# def pca_app(mode:str, 
#             n_components:int = 4,
#             tooltip_alpha: float = 0.75,
#             img_alpha: float = 0.7,
#             width: float = 150,
#             ):
    
    
#     fitted_dataaset = fit_pca(n_components=n_components)
#     fig = generate_pca_plot(fitted_dataset)
    
#     # select axes based on user choice? 
#     # https://stackoverflow.com/questions/54583996/how-to-use-callback-to-update-bar-graph-in-dash
#     if mode == 'inline':
#         app = JupyterDash(__name__)
#     else:
#         app = Dash(__name__)
        
#     app.layout = html.Div([
#         dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
#         dcc.Tooltip(
#                 id="graph-tooltip", background_color=f"rgba(255,255,255,{tooltip_alpha})"
#             ),
#         ])
    
#     @app.callback(
#         output=[
#             Output("graph-tooltip", "show"),
#             Output("graph-tooltip", "bbox"),
#             Output("graph-tooltip", "children"),
#         ],
#         inputs=[Input("graph-basic-2", "hoverData"),
#                 ],
#     )
#     def display_hover(hoverData, value):
#         if hoverData is None:
#             return False, no_update, no_update
        
#         pt = hoverData["points"][0]
#         bbox = pt["bbox"]
#         num = pt["pointNumber"]
#         curve_num = pt["curveNumber"]
        
#         if len(fig.data) != 1:
#             df_curve = curve_dict[curve_num].reset_index(drop=True)
#             df_row = df_curve.iloc[num]
#         else:
#             df_row = df.iloc[num]
        
#         img_str = 
#         hoverbox_elements = [html.Img(
#                         src=img_str,
#                         style={
#                             "width": "100%",
#                             "background-color": f"rgba(255,255,255,{img_alpha})",
#                         },
#                     )]
#         children = [
#             html.Div(
#                 hoverbox_elements,
#                 style={
#                     "width": f"{width}px",
#                     "white-space": "normal",
#                 },
#             )
#         ]
#         return True, bbox, children
    
#     # define mouseover pillow image
#     return app
#%%

def search_app(mode:str):
    
    library_dataframe = load_case_study_dataframe()
    
    image_model = load_image_model()
    transform = return_default_transforms()
    
    if mode == 'inline':
        app = JupyterDash(__name__)
    else:
        app = Dash(__name__)
        
    app.layout = html.Div([
        html.H1('Search the Cambridge University Digital Library for Images!'),
        dcc.Upload(
            id='upload-image',
            children = html.Button('Upload File')),
        dcc.Slider(1, 100,
            id='topN-slider',
            value=5,
    ),
        html.Div(id='output-image-upload'),
        html.Div(id='search-results', style= {'display':'inline-block'}),
        
    ], style= {'display':'inline-block'})
    
    def parse_contents(contents, filename, date):
        img_content = html.Div([

            html.H5('Uploaded Image:'),
            html.Img(src=contents, style={'height':'400px'}),
            html.Hr(),
            # html.Div('Raw Content'),
            # html.Pre(contents[0:200] + '...', style={
            #     'whiteSpace': 'pre-wrap',
            #     'wordBreak': 'break-all'
            # })
            ], style={'display': 'inline-block'})
        return img_content
    
    def return_search_results(img_vector, top_n:int):
        results = perform_top_n_search(img_vector, search_dataframe=library_dataframe, top_n=top_n)
        
        search_content = [html.H5('Most similar images:')]
        for _, row in results.iterrows():

            hit_content = html.Div([html.Img(src=row['img-url'], style={"height": "400px"}),
                           html.P(f'{row["item-label"]} - {row["img-label"]}'),
                           html.P(f'Similarity: {row["similarity"]:.4f}')], style={'display': 'inline-block'})
            # for item in hit_content:
            search_content.append(hit_content)
            
        search_content = html.Div(search_content, style={'display': 'inline-block'})

        return search_content
    
    @app.callback(Output('output-image-upload', 'children'),
                Input('upload-image', 'contents'),
                State('upload-image', 'filename'),
                State('upload-image', 'last_modified'))
    def display_uploaded_image(contents, filename, last_modified):
        if contents is not None:
            children = parse_contents(contents, filename, last_modified)
            return children
        
    @app.callback(Output('search-results', 'children'),
                Input('upload-image', 'contents'),
                Input('topN-slider', 'value'),)
    def display_search_results(contents, value):
        if contents is not None:
            pil_image = pil_image_from_html_img_src(contents)
            vector = generate_vector_for_pil_image(pil_image=pil_image, image_model=image_model, torchvision_transform=transform)
            children = return_search_results(img_vector=vector, top_n=value)
            return children
    return app

def pil_image_from_html_img_src(img_src):
    img_string = img_src.split(',')[1]
    im_bytes = base64.b64decode(img_string)   # im_bytes is a binary image
    im_file = io.BytesIO(im_bytes)
    pil_image = Image.open(im_file).convert("RGB")
    return pil_image

def run_search_app(mode:str = 'inline', port: int = 8701, height:int = 1000):
    dash_app = search_app(mode)

    dash_app.run_server(mode=mode, port=port, height=height)
    return
run_search_app()

#%%
if __name__ == '__main__':
    Fire(run_search_app)