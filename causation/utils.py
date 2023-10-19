""" utils.py

Set of utilities used for the causation project.
"""
import os
from pathlib import Path
import logging

import openai
import panel as pn
from openai.error import AuthenticationError

logger = logging.getLogger(__name__)

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
        # dir_ = Path(mkdtemp())
        dir_ = Path(".")  # for binder, allow editing of uploaded file via lab interface.
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
        if len(key) == 0:
            return "Please enter your OpenAI API key."
        else:
            if len(key) == 51:
                try:
                    openai.api_key = key
                    _ = openai.Model.list()
                    os.environ['OPENAI_API_KEY'] = key
                    return "Valid API Key. Please continue."
                except AuthenticationError as ae:
                    return str(ae)
                except Exception as e:
                    logger.debug(str(e))
                    return "Something went wrong when validating API Key. Please try again."
            return "Incorrect API key provided. Must be 51 characters."

    iobject = pn.bind(_cb_overwrite_api, password_input.param.value, watch=False)
    return pn.Row(password_input, pn.pane.Markdown(iobject))
