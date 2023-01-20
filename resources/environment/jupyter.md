# Jupyter Lab 

The site is powered by a [JupyterLab Lite](https://jupyterlite.readthedocs.io/en/latest/index.html) deployed  

`https://pantelis.github.io/data-science-repl/`

that is primarily used to run the REPL code fragments in the course notes.  

A Python REPL can be launched by including the following snippet in a markdown file:

```{raw} html
<iframe
  src="https://pantelis.github.io/data-science-repl/repl/index.html?theme=JupyterLab Dark&toolbar=1&kernel=python&code=import numpy as np"
  width="100%"
  height="100%"
></iframe>
```

:::{note}
Try it by entering the following code in the REPL:

```python
3+4
```
and then pressing the play button. 

::::