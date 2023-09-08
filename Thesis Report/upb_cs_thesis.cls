%This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
%%%
% LaTeX class for documents submitted to UPB's CS department's research groups
% This is just a guideline - modify as you like :)
% Last Update: 2017-12-18
% This file is based upon the LaTeX template used by AG Blömer.
% The file and template were created by Nils Löken and submitted to UPB CS's First Thesis Template Tournament.
%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{upb_cs_thesis}[2017/12/18 initial version]

\newif\ifgerman\germanfalse
\DeclareOption{german}{\germantrue}

\ProcessOptions
\LoadClass[a4paper, 11pt, twoside, openright]{book}
\ifgerman
	\RequirePackage[ngerman]{babel}
\else
	\RequirePackage[english]{babel}
\fi
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage{csquotes}
\RequirePackage{tikz}
\RequirePackage{calc}
\RequirePackage[explicit]{titlesec}

\reversemarginpar
\geometry{a4paper, twoside, left=30mm, right=20mm, top=30mm, bottom=30mm}

\pagestyle{empty}
\pagenumbering{roman}


\newcommand{\@upbchaptername}{}
\newcommand{\@upbsectionname}{}
\newcommand{\ps@upb}{%
	\renewcommand{\@oddhead}{\leftmark\hfill}%
	\renewcommand{\@evenhead}{\hfill\rightmark}%
	\renewcommand{\@oddfoot}{\hfill\thepage\hfill}%
	\renewcommand{\@evenfoot}{\hfill\thepage\hfill}%
}


\newcommand*{\emptypage}{\newpage\null\thispagestyle{empty}\newpage}


\newcommand{\thetitle}{undefined}
\newcommand{\theauthor}{undefined}
\let\oldtitle=\title
\let\oldauthor=\author
\renewcommand{\title}[1]{\renewcommand{\thetitle}{#1}\oldtitle{#1}}
\renewcommand{\author}[1]{\renewcommand{\theauthor}{#1}\oldauthor{#1}}
\newcommand{\thethesistype}{undefined}
\newcommand{\thedegree}{undefined}
\newcommand{\theresearchgroup}{undefined}
\newcommand{\thesupervisor}{undefined}
\newcommand{\thesubmissiondate}{undefined}
\newcommand{\thesistype}[1]{\renewcommand{\thethesistype}{#1}}
\newcommand{\degree}[1]{\renewcommand{\thedegree}{#1}}
\newcommand{\researchgroup}[1]{\renewcommand{\theresearchgroup}{#1}}
\newcommand{\supervisor}[1]{\renewcommand{\thesupervisor}{#1}}
\newcommand{\submissiondate}[1]{\renewcommand{\thesubmissiondate}{#1}}


\newenvironment{abstract}{
	\begin{center}
		\begin{minipage}{.9\textwidth}
			\ifgerman
				\textbf{Zusammenfassung.} \hspace*{0.10pt}
			\else
				\textbf{Abstract.} \hspace*{0.10pt}
			\fi
}{
		\end{minipage}
	\end{center}
}


\let\@upbtocold=\tableofcontents
\renewcommand{\tableofcontents}{
	\pagestyle{plain}
	\@upbtocold
	\cleardoublepage
	\setcounter{page}{0}
	\pagestyle{upb}
	\pagenumbering{arabic}
	\renewcommand{\chaptermark}[1]{
		\markboth{\textsc{\chaptername\ \thechapter.\ ##1}}{}
	}
	\renewcommand{\sectionmark}[1]{
		\markright{\textsc{\thesection\ ##1}}{}
	}
}


\newlength{\@upbchapternumberwidth}
\newlength{\@upbtitlemaxwidth}
\newlength{\@upbtitletextwidth}

\titleformat{\chapter}{
	\normalfont\huge\bfseries
}{}{0em}{
	\newcommand*{\@upbprintscaledchapternumber}{
		\resizebox{!}{15mm}{\thechapter}
	}
	\settowidth{\@upbchapternumberwidth}{
		\@upbprintscaledchapternumber
	}
	\setlength{\@upbtitlemaxwidth}{
		.9\textwidth - \@upbchapternumberwidth 
	}
	\settowidth{\@upbtitletextwidth}{#1}
	\flushright
	\rlap{
		\parbox[b]{\textwidth + 23mm}{
			\parbox[b]{\@upbtitlemaxwidth}{
				\ifdim\@upbtitletextwidth<\@upbtitlemaxwidth%
					\hfill #1
				\else
					#1\hbadness=10000
				\fi
			}
			\hspace{12mm}
			\makebox[\@upbchapternumberwidth][b]{
				\raisebox{12mm}{\@upbprintscaledchapternumber}
			}
			\hspace{-\@upbchapternumberwidth}
			\hspace{-12mm}
			\begin{tikzpicture}
				\fill[black] (0mm, 0mm) rectangle
						(\@upbchapternumberwidth +23mm, 10mm);
			\end{tikzpicture}
		}
	}
}

\titleformat{name=\chapter,numberless}{
	\normalfont\huge\bfseries
}{}{0em}{
	\settowidth{\@upbchapternumberwidth}{
		\resizebox{!}{15mm}{0}
	}
	\setlength{\@upbtitlemaxwidth}{
		.9\textwidth - \@upbchapternumberwidth 
	}
	\settowidth{\@upbtitletextwidth}{#1}
	\flushright
	\rlap{
		\parbox[b]{\textwidth + 23mm}{
			\parbox[b]{\@upbtitlemaxwidth}{
				\ifdim\@upbtitletextwidth<\@upbtitlemaxwidth
					\hfill #1
				\else
					#1\hbadness=10000
				\fi
			}
			\hspace{8mm}
			\begin{tikzpicture}
				\fill[black] (0mm, 0mm) rectangle
						(\@upbchapternumberwidth +23mm, 10mm);
			\end{tikzpicture}
		}
	}
}

\titlespacing*{\chapter}{0pt}{30pt}{25pt} 

