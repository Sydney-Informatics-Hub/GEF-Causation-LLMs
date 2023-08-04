""" utils.py

Set of utilities used for the causation project.
"""
import os
import panel as pn
from tempfile import mkdtemp
from pathlib import Path

MARKDOWN = str


def fileuploader(ext: str) -> (pn.Row, dict):
    """ Creates a file uploader specifically to be used by CoT.
    :returns widget, data cache as dict.
    """
    pn.extension()
    finput = pn.widgets.FileInput(accept=ext)
    uploader_data = dict(data=None)

    def _cb_save_to_file(fbytes: bytes, fname: str) -> MARKDOWN:
        if fbytes is None or len(fbytes) <= 0 or fname is None: return ""
        dir_ = Path(mkdtemp())
        path = dir_.joinpath(fname)
        with open(path, 'wb') as h:
            h.write(finput.value)
        uploader_data['data'] = path
        return f"Saved **{fname}**."

    iobject = pn.bind(_cb_save_to_file, finput, finput.param.filename)
    return pn.Row(finput, pn.pane.Markdown(iobject)), uploader_data


def openai_apikey_input():
    pn.extension()
    password_input = pn.widgets.PasswordInput(name='Enter your OpenAI API Key (then press enter):',
                                              placeholder='<OpenAI API Key>')

    def _cb_overwrite_api(key: str):
        os.environ['OPENAI_API_KEY'] = key
        if len(os.environ['OPENAI_API_KEY']) == 51:
            return "Valid API Key. Please continue."
        elif len(os.environ['OPENAI_API_KEY']) == 0:
            return "Please enter your OpenAI API Key."
        else:
            return "Invalid API Key."

    iobject = pn.bind(_cb_overwrite_api, password_input.param.value, watch=False)
    return pn.Row(password_input, pn.pane.Markdown(iobject))
