from pygments.style import Style
from pygments.token import Error, Text, Whitespace, Other


class MyStyle(Style):
    background_color = "black"
    highlight_color = "#49483e"

    styles = {
        Text:                      "#f8f8f2", # class:  ''
        Whitespace:                "",        # class: 'w'
        Error:                     "#960050 bg:#1e0010", # class: 'err'
        Other:                     "",        # class 'x'
    }
