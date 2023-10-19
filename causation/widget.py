""" widget.py

This contains the full classification widget.

note: class name Controller is misleading - it does everything.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import panel as pn
from panel.theme import Material
from panel.widgets import (
    FileInput, Select, TooltipIcon, FloatSlider, DataFrame
)
from panel.pane import (
    HTML
)
from panel.layout import (
    Row, Column, Accordion
)

pn.extension()
pn.config.design = Material

MARKDOWN = str


class FileUploadToDir(object):
    def __init__(self, ext: list[str], upload_dir: str):
        self._extensions = ext
        upload_dir = Path(upload_dir)
        if upload_dir.is_file(): raise FileExistsError(f"{upload_dir} is a file.")
        upload_dir.mkdir(exist_ok=True)
        self._upload_dir = upload_dir

        self._uploaded = dict()
        self._finput = FileInput(accept=",".join(ext))

        iobj = pn.bind(self._cb_save_to_file, self._finput, self._finput.param.filename)
        self._widget = pn.Row(self._finput, pn.pane.Markdown(iobj))

    @property
    def ext(self) -> list[str]:
        return self._extensions.copy()

    @property
    def upload_dir(self) -> Path:
        return self._upload_dir

    def uploaded(self) -> Optional[Path]:
        uploaded = self._uploaded.get('file_path')
        return Path(uploaded) if uploaded is not None else None

    def _cb_save_to_file(self, fbytes: bytes, fname: str) -> MARKDOWN:
        if fbytes is None or len(fbytes) <= 0 or fname is None: return ""
        # dir_ = Path(mkdtemp())
        dir_ = Path(self._upload_dir)  # for binder, allow editing of uploaded file via lab interface.
        path = dir_.joinpath(fname)
        with open(path, 'wb') as h:
            h.write(self._finput.value)
        self._uploaded['file_path'] = path
        return f"Saved **{fname}**."

    def widget(self):
        return self._widget


class DatasetUpload(FileUploadToDir):
    def __init__(self, required_cols: str | list[str], upload_dir: str, ):
        super().__init__(ext=[".xlsx", ".csv"], upload_dir=upload_dir)

        self._required_cols = required_cols if isinstance(required_cols, list) else [required_cols]

        # note: bind args must be exact as FileUploadToDir for sequential behaviour.
        obj2 = pn.bind(self._on_upload_display, self._finput, self._finput.param.filename)
        self._widget = Column(self._widget, obj2, sizing_mode='stretch_width')

    def _on_upload_display(self, _: bytes, __: str) -> pn.pane.Str | pn.widgets.DataFrame | None:
        uploaded: Optional[Path] = self.uploaded()
        if uploaded is None: return None

        if uploaded.suffix == ".csv":
            df = pd.read_csv(uploaded)
        elif uploaded.suffix == ".xlsx":
            df = pd.read_excel(uploaded)
        else:
            raise NotImplementedError(f"Only {self.ext} are supported. This should've been blocked at widget level.")
        try:
            self._validate_required_cols(df)
        except ValueError as ve:
            return pn.pane.Str(ve)
        return DataFrame(df, show_index=False, height=400, sizing_mode='stretch_width', disabled=True, )

    def _validate_required_cols(self, df: pd.DataFrame):
        missing = list()
        for rc in self._required_cols:
            if rc not in df.columns:
                missing.append(rc)
        if len(missing) > 0:
            raise ValueError(f"Required columns: {str(', '.join(missing))} not found in {df.columns}.")


class CoTPromptUpload(FileUploadToDir):
    def __init__(self, upload_dir: str):
        super().__init__(ext=[".toml"], upload_dir=upload_dir)

        # note: bind args must be exact as FileUploadToDir for sequential behaviour.
        obj2 = pn.bind(self._on_upload_display, self._finput, self._finput.param.filename)
        self._widget = Column(self._widget, obj2, sizing_mode='stretch_width')

    def _on_upload_display(self, _: bytes, __: str) -> pn.Accordion | None:
        uploaded: Optional[Path] = self.uploaded()
        if uploaded is None: return None
        if uploaded.suffix == ".toml":
            return self.from_toml_to_accordian(uploaded)
        else:
            raise NotImplementedError(f"Only {self.ext} are supported. This should've been blocked at widget level.")

    @staticmethod
    def from_toml_to_accordian(path: Path):
        # todo: placeholder only
        df = pd.DataFrame([f'query {i}' for i in range(100)], columns=['sentence'])
        instructions = """
        You are an expert in philosophy, spefically in the domain of biases.
        You are trying to identify biases that deterministically or categorically link genes to "traits" or "phenotypes", including obese, obesity, overweight, diabetic, diabetes, heavy, fat, fatness; and also behaviours such as eating, overeating, hunger, hungry, craving, fat storage, weight gain, gaining weight, weight loss, losing weight, exercise, physical activity, burning calories.\
        Sentence must be about genes causing a trait, not about a trait affecting a gene.
        When genes "do not", "don't", or "didn't" cause or determine or lead to a trait, then no bias. 
        """

        html_str = """
        <h3>Instructions</h3>
        {instructions}
        """
        return pn.Accordion(('class',
                             pn.Column(pn.pane.HTML(html_str.format(instructions=instructions).lstrip()),
                                       pn.pane.DataFrame(df, index=False, height=200, sizing_mode='stretch_width'))))


class ModelConfig(object):
    def __init__(self):
        # model selector
        model = pn.widgets.Select(options=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
        model_name_ttip = pn.widgets.TooltipIcon(value="This is a simple tooltip by using a string",
                                                 margin=(-33, -500, 20, -170))
        # top p slider
        top_p = pn.widgets.FloatSlider(name="Top p", start=0.1, end=1.0, step=0.1, value=0.8, tooltips=True)
        top_p_ttip = pn.widgets.TooltipIcon(value="This is a simple tooltip by using a string",
                                            margin=(-43, -40, 30, -170))
        # temperature slider
        temp = pn.widgets.FloatSlider(name="Temperature", start=0.0, end=2.0, step=0.1, value=1.0, tooltips=False)
        temp_ttip = pn.widgets.TooltipIcon(value="This is a simple tooltip by using a string",
                                           margin=(-43, -120, 50, -170))

        mconfig = pn.Column("## Model Configuration",
                            model, model_name_ttip,
                            top_p, top_p_ttip,
                            temp, temp_ttip)
        self._widget = mconfig

        self._config = dict(
            model=model.value,
            top_p=top_p.value,
            temperature=temp.value
        )
        model.param.watch(self._cb_on_model_select, 'value')
        top_p.param.watch(self._cb_on_top_p_slide, 'value')
        temp.param.watch(self._cb_on_temp_slide, 'value')

    def _cb_on_model_select(self, event):
        self._config['model'] = event.obj.value

    def _cb_on_top_p_slide(self, event):
        self._config['top_p'] = event.obj.value

    def _cb_on_temp_slide(self, event):
        self._config['temperature'] = event.obj.value


class Controller(object):
    pass
