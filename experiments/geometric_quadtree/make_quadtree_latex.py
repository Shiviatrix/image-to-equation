import os
import subprocess

def create_latex():
    # Read the full AST generated earlier.
    with open("scratch/einstein_quadtree_ast.txt", "r") as f:
        ast_str = f.read().strip()
        
    # To prevent LaTeX from crashing on a 200,000 character line,
    # we manually break it into lines of 100 characters.
    chunk_size = 100
    ast_lines = [ast_str[i:i+chunk_size] for i in range(0, len(ast_str), chunk_size)]
    ast_formatted = "\n".join(ast_lines)
    
    latex_content = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{verbatim}
\geometry{a4paper, margin=0.5in}

\begin{document}

\begin{center}
    \textbf{\Large The QuadTree Mosaic: Formal Mathematical Representation}
    
    \vspace{0.5cm}
    \includegraphics[width=0.6\textwidth]{einstein_quadtree.png}
\end{center}

\vspace{0.5cm}
\noindent \textbf{Mathematical Formulation (Procedural Spatial Partitioning)}

\vspace{0.2cm}
\noindent The image above is generated entirely by the following procedural function, consisting of 3,460 geometric constraints (boxes):
\begin{equation*}
    F(x,y) = \sum_{k=1}^{3460} C_k \cdot \text{sdfmask}\left( \text{box}\left(x, y, cx_k, cy_k, w_k, h_k\right), 0.001 \right)
\end{equation*}

\noindent Where:
\begin{itemize}
    \item $C_k$ is the localized brightness scalar of the $k$-th quadtree cell.
    \item $cx_k, cy_k$ are the precise spatial coordinates of the center of the cell.
    \item $w_k, h_k$ define the scale/depth of the cell in the QuadTree hierarchy.
    \item $\text{sdfmask}(\dots, 0.001)$ serves as a mathematically continuous boolean intersection, returning 1 inside the cell and 0 outside.
\end{itemize}

\vspace{0.5cm}
\noindent \textbf{Complete QuadTree AST Source Code (All 200,748 characters):}

\scriptsize
\begin{verbatim}
""" + ast_formatted + r"""
\end{verbatim}
\normalsize

\end{document}
"""

    with open("scratch/einstein_quadtree.tex", "w") as f:
        f.write(latex_content)
        
    print("Compiling LaTeX...")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "einstein_quadtree.tex"], cwd="scratch/")
    
    print("Moving PDF to workspace root...")
    subprocess.run(["cp", "scratch/einstein_quadtree.pdf", "einstein_quadtree_equation.pdf"])
    print("Done!")

if __name__ == "__main__":
    create_latex()
