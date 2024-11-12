from fasthtml.common import *

app,rt = fast_app(hdrs=[HighlightJS()])

@rt("/")
def get():
    return Titled("😄 Demo",
            Main(
                H3("Hello World 🌎"),
                P("Esto es un texto de ejemplo")),
                A("Ir a Google", href="https://www.google.com"),
                Hr(),
                A('Ir a about', href='/about'),
            Footer("© 2021"))
@rt("/about")
def about():
    return Main(
    H3('Inline text elements'),
    Div(
        P(
            A('Primary link', href='#', onclick='event.preventDefault()')
        ),
        P(
            A('Secondary link', href='#', onclick='event.preventDefault()', cls='secondary')
        ),
        P(
            A('Contrast link', href='#', onclick='event.preventDefault()', cls='contrast')
        ),
        cls='grid'
    )
)

app,rt = fast_app(hdrs=[HighlightJS()])

@rt("/convert")
def post(html:str, attr1st:bool): return Pre(Code(html2ft(html, attr1st=str2bool(attr1st)))) if html else ''

@rt("/")
def get():
    return Titled(
        "Convert HTML to FT",
        Form(hx_post='/convert', target_id="ft", hx_trigger="change from:#attr1st, keyup delay:500ms from:#html")(
            Select(style="width: auto", id="attr1st")(
                Option("Children 1st", value="0", selected=True), Option("Attrs 1st", value="1")),
            Textarea(placeholder='Paste HTML here', id="html", rows=10)),
        Div(id="ft"))

serve()
serve()
