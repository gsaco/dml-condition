\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ectj}
\newcounter{ectj@status}
\setcounter{ectj@status}{0}
 \DeclareOption{draft}{\setlength\overfullrule{5pt}\setcounter{ectj@status}{0}}
 \DeclareOption{proof}{\setlength\overfullrule{5pt}\setcounter{ectj@status}{1}}
 \DeclareOption{crc}{\setlength\overfullrule{0pt}\setcounter{ectj@status}{2}}
 \DeclareOption{final}{\setlength\overfullrule{0pt}\setcounter{ectj@status}{3}}
\newif\if@@pdf
\@@pdffalse
 \DeclareOption{pdf}{\setlength\overfullrule{0pt}\@@pdftrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{mathptm}{%
 \def\ECTJ@fonts{\RequirePackage{times,mathptm}}%
 \renewcommand{\ttdefault}{cmtt}}
\DeclareOption{mathtime}{%
 \def\ECTJ@fonts{\RequirePackage{times}\RequirePackage%
[nofontinfo,mtbold]%
{mathtime}%
 \renewcommand{\ttdefault}{cmtt}%
%\DeclareMathSymbol\dagger {\mathbin}{operators}{160}%
%\DeclareMathSymbol\ddagger{\mathbin}{operators}{224}%
%\DeclareMathSymbol\mathsection{\mathord}{operators}{164}%
%\DeclareMathSymbol\mathparagraph{\mathord}{operators}{166}%
%\DeclareMathSymbol\mathsterling{\mathord}{operators}{163}%
}}
\DeclareOption{cmfonts}{%
 \let\ECTJ@fonts\relax}
\let\UPPERCASE\sc
\def\Uppercase#1{\UPPERCASE#1}
\font\UPPERCASE=cmcsc10
\ExecuteOptions{cmfonts,draft}
  \setlength\paperheight {297mm}
  \setlength\paperwidth {210mm}
\ProcessOptions
\ECTJ@fonts
\RequirePackage[final]{graphicx}%options changes from final to emtex
\DeclareGraphicsExtensions{eps}
\RequirePackage{color}
\RequirePackage{ifthen}
\RequirePackage{latexsym}
\def\apdflink#1#2{#2}
\if@@pdf
 \RequirePackage{pdfmark}
 \let\link=\pdf@link
 \let\dest=\pdf@dest
 \let\query\pdf@note
 \ifnum\theectj@status=2
  \def\query#1#2{#2}
 \else
 \ifnum\theectj@status=3
  \def\query#1#2{#2}
 \fi\fi
\else
\ifnum\theectj@status=0
 \def\link#1#2{#2\rlap{\smash{\textsuperscript{{\color{red}{\texttt{#1}}}}}}}
 \def\dest#1#2{}
 \def\query#1#2{\marginpar{{\color{green}\fontsize{6.5pt}{6.5pt}\itshape\ttfamily
              #1\vss\color{black}}}{\color{green}#2\color{black}}}
\else
\ifnum\theectj@status=1
 \def\link#1#2{#2}
 \def\dest#1#2{}
 \def\query#1#2{\marginpar{{\color{green}\fontsize{6.5pt}{6.5pt}\itshape\ttfamily
              #1\vss\color{black}}}{\color{green}#2\color{black}}}
\else
\ifnum\theectj@status=2
 \def\link#1#2{#2}
 \def\dest#1#2{}
 \def\query#1#2{#2}
\else
\ifnum\theectj@status=3
 \def\link#1#2{#2}
 \def\dest#1#2{}
 \def\query#1#2{#2}
\fi\fi\fi\fi
\fi
\DeclareMathAlphabet\mathbi
 \encodingdefault\rmdefault\bfdefault\itdefault
\DeclareMathAlphabet\mathsc
 \encodingdefault{cmr}\mddefault\scdefault
\renewcommand\normalsize{%
  \@setfontsize\normalsize\@xpt\@xiipt
  \abovedisplayskip 6\p@ \@plus1\p@ \@minus1\p@
  \abovedisplayshortskip 3\p@ \@plus1\p@
  \belowdisplayshortskip 3\p@ \@plus1\p@
  \belowdisplayskip \abovedisplayskip
  \let\@listi\@listI}
\normalsize
\newcommand\small{%
  \@setfontsize\small\@ixpt{10}%
  \abovedisplayskip 5\p@ \@plus1\p@ \@minus1\p@
  \abovedisplayshortskip 2.5\z@ \@plus1\p@
  \belowdisplayshortskip 2.5\p@ \@plus1\p@
  \def\@listi{\leftmargin\leftmargini
        \topsep 10\p@ \@plus1\p@ \@minus1\p@
        \parsep \z@
        \itemsep \parsep}%
  \belowdisplayskip \abovedisplayskip
}
\newcommand\mediumsize{%
  \@setfontsize\small\@viiipt{10}%
  \abovedisplayskip 5\p@ \@plus1\p@ \@minus1\p@
  \abovedisplayshortskip 2.5\z@ \@plus1\p@
  \belowdisplayshortskip 3\z@ \@plus3\p@ \@minus1\p@
  \def\@listi{\leftmargin\leftmargini
        \topsep 10\p@ \@plus1\p@ \@minus1\p@
        \parsep \z@
        \itemsep \parsep}%
  \belowdisplayskip \abovedisplayskip
}
\newcommand\footnotesize{%
  \@setfontsize\footnotesize\@viiipt{9}%
  \abovedisplayskip 4.5\p@ \@plus1\p@ \@minus1\p@
  \abovedisplayshortskip \z@ \@plus1\p@
  \belowdisplayshortskip \z@ \@plus1\p@
  \def\@listi{\leftmargin\leftmargini
        \topsep 9\p@ \@plus1\p@ \@minus1\p@
        \parsep \z@
        \itemsep \parsep}%
  \belowdisplayskip \abovedisplayskip
}
\newcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\newcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\newcommand\large{\@setfontsize\large\@xiipt{14}}
\newcommand\Large{\@setfontsize\Large\@xivpt{16}}
\newcommand\LARGE{\@setfontsize\LARGE\@xviipt{21}}
\newcommand\huge{\@setfontsize\huge\@xxpt{25}}
\newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{1}
\setlength\parskip{0\p@ \@plus 0.1\p@}
\setlength\parindent{1em}
\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
\@lowpenalty  51
\@medpenalty 151
\@highpenalty 301
\clubpenalty 0
\widowpenalty 10000
\@twosidetrue
\@mparswitchtrue
\setlength\headheight{18\p@}
\setlength\headsep  { 6\p@}
\setlength\topskip  {12\p@}
\setlength\footskip{18\p@}
\setlength\maxdepth{0.5\topskip}
\setlength\textwidth{32pc}
 \setlength\@tempdima{\paperwidth}
 \addtolength\@tempdima{-2in}
 \setlength\@tempdimb{32pc}
  \ifdim\@tempdima>\@tempdimb\relax
   \setlength\textwidth{\@tempdimb}
  \else
   \setlength\textwidth{\@tempdima}
  \fi
 \@settopoint\textwidth
\setlength\textheight{47\baselineskip}
\addtolength\textheight{\topskip}
 \@settopoint\textheight
\setlength\marginparsep{10\p@}
\setlength\marginparpush{6\p@}
  \setlength\@tempdima    {\paperwidth}
  \addtolength\@tempdima   {-\textwidth}
  \setlength\oddsidemargin  {.5\@tempdima}
  \addtolength\oddsidemargin {-1in}
  \setlength\marginparwidth  {.5\@tempdima}
  \addtolength\marginparwidth {-\marginparsep}
  \addtolength\marginparwidth {-0.4in}
  \addtolength\marginparwidth {-.4in}
 \ifdim \marginparwidth >2in
   \setlength\marginparwidth{2in}
 \fi
 \setlength\marginparwidth{8pc}
 \@settopoint\oddsidemargin
 \@settopoint\marginparwidth
 \setlength\evensidemargin {\paperwidth}
 \addtolength\evensidemargin{-2in}
 \addtolength\evensidemargin{-\textwidth}
 \addtolength\evensidemargin{-\oddsidemargin}
 \@settopoint\evensidemargin
 \setlength\topmargin{\paperheight}
 \addtolength\topmargin{-2in}
 \addtolength\topmargin{-\headheight}
 \addtolength\topmargin{-\headsep}
 \addtolength\topmargin{-\textheight}
 \addtolength\topmargin{-\footskip}   % this might be wrong!
 \addtolength\topmargin{-0.5\topmargin}
 \@settopoint\topmargin
\setlength\footnotesep{6\p@}
\setlength{\skip\footins}{12\p@ \@plus 12\p@ \@minus 1\p@}
\setcounter{topnumber}{3}
\renewcommand\topfraction{1}
\setcounter{bottomnumber}{2}
\renewcommand\bottomfraction{0.8}
\setcounter{totalnumber}{4}
\renewcommand\textfraction{0}
\renewcommand\floatpagefraction{0.8}
\setlength\floatsep  {12\p@ \@plus 6\p@ \@minus 1\p@}
\setlength\textfloatsep{18\p@ \@plus 6\p@ \@minus 3\p@}
\setlength\intextsep  {18\p@ \@plus 3\p@ \@minus 2\p@}
\setlength\@fptop{0\p@ \@plus 0fil}
\setlength\@fpsep{12\p@ \@plus 0fil}
\setlength\@fpbot{0\p@ \@plus 3fil}
\let\@accepted\@empty
\def\accepted#1{\gdef\@accepted{#1}}
\let\@received\@empty
\def\received#1{\gdef\@received{#1}}
\let\@revised\@empty
\def\revised#1{\gdef\@revised{#1}}
\def\volume#1{\setcounter{@volumenumber}{#1}}
 \newcounter{@volumenumber}
 \setcounter{@volumenumber}{1}
\def\issue#1{\setcounter{@issuenumber}{#1}}
 \newcounter{@issuenumber}
 \setcounter{@issuenumber}{1}
\def\four@digits#1{\ifnum#1<1000 0\fi\ifnum#1<100 0\fi\ifnum#1<10 0\fi\number#1}
\def\startpage{\@ifundefined{r@firstpage}{????}%
  {\expandafter\four@digits{\expandafter\@secondoftwo\r@firstpage}}}
\def\pages{\@ifundefined{r@lastpage}{??}%
 {\addtocounter{@pages}{\expandafter\@secondoftwo\r@lastpage}%
  \addtocounter{@pages}{-\expandafter\@secondoftwo\r@firstpage}%
  \two@digits{\value{@pages}}}}
 \newcounter{@pages}
 \setcounter{@pages}{1}
\def\artid#1{\gdef\@artid{#1}}
\def\journalname#1{\gdef\@journalname{#1}}
\def\journalnumber#1{\gdef\@journalnumber{#1}}
\def\copyrightname#1{\gdef\@copyrightname{#1}}
\def\publname#1{\gdef\@publname{#1}}
\def\publadd#1{\gdef\@publadd{#1}}
\def\price#1{\gdef\@price{#1}}
\volume{1}
\issue{1}
\artid{ectj??????}
\journalname{The Econometrics Journal}
\journalnumber{0747--7171}
\copyrightname{Royal Economic Society}
\publname{Blackwell Publishers Ltd}
\publadd{108~Cowley Road, Oxford OX4 1JF, UK and 350 Main Street,%
Malden, MA, 02148, USA.}
\price{\$30.00}
\def\label@firstpage{%
  \immediate\write\@auxout{\string
  \bibstyle{chicago}}%
  \immediate\write\@auxout{\string
  \newlabel{firstpage}{{}{\thepage}}}%
}
\def\label@lastpage{%
 \advance\c@page\m@ne
  \immediate\write\@auxout{\string
  \newlabel{lastpage}{{}{\thepage}}}%
  \advance\c@page\@ne}
\def\ps@ectj{%
 \def\@oddhead{%
   \relax
   {\small\hfil{\itshape\@shorttitle}\hfil\thepage
   }}%
 \def\@evenhead{%
   {\small\thepage\hfil{\itshape\@shortauthors}\hfil
   }}%
 \def\@oddfoot{}
 \def\@evenfoot{}
 }
\ps@ectj
\def\ps@title{%
 \def\@oddhead{%
   {\smash{\lower2pt\rlap{{}}}\raise8pt\vbox{\hsize\textwidth\@parboxrestore\small
     {\rmfamily\small{Manuscript submitted to }{\itshape\@journalname\/}, %
     \space pp.~\pageref{firstpage}--\pageref{lastpage}.}}}%
   \relax
   \relax}%
 \let\@evenhead\@oddhead
 \def\@oddfoot{%
  {\vbox{\vspace{.5\baselineskip}\hsize\textwidth\@parboxrestore\footnotesize
     { }}}%
   \relax
   }%
 }
 \def\ps@empty{%
   \let\@oddfoot\@empty\let\@evenfoot\@empty
   \let\@evenhead@empty\let\@oddhead\@empty
 }
\def\affiliation#1{#1}
\let\@address\@empty
\let\@email\@empty

\def\email#1{\g@addto@macro\@authors{\\{\rm E-mail:} {\tt #1}}}

\def\title{\@dblarg\ECTJ@title}
\def\ECTJ@title[#1]#2{%
 \def\@title{#2}%
 \def\@shorttitle{#1}}
\newif\if@@address
\@@addressfalse
\def\address{\@dblarg\ECTJ@address}
\def\ECTJ@address[#1]#2{%
\if@@address
\g@addto@macro\@authors{\\{%
\small\textit{\begin{tabular}{@{}c@{}}#2\end{tabular}}}}
\else
\g@addto@macro\@authors{\\[8pt]{%
\small\textit{\begin{tabular}{@{}c@{}}#2\end{tabular}}}}
\fi\global\@@addresstrue
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\author{\@dblarg\ECTJ@author}
\def\authorsetal{\relax}
\let\authors\author
\def\ECTJ@author[#1]#2{%
 \ifx\@authors\@empty
  \def\@shortauthors{#1}
  \def\@authors{{\UPPERCASE#2}}
 \else
\if@@address
  \g@addto@macro\@authors{%
  \\\hbox{~}\\{\UPPERCASE#2}}%
\else%
  \g@addto@macro\@authors{%
  \\{\UPPERCASE#2}}%
\fi%  \ECTJ@addshortauthor{, #1}
 \fi\global\@@addressfalse}
\def\ECTJ@addshortauthor#1{%
 \g@addto@macro\@shortauthors{#1}}
\def\etalname{\textit{et~al}.}
\let\@authors\@empty
\let\@shortauthors\@authors


\newcommand\maketitle{%
 \thispagestyle{title}\label@firstpage
 \let\footnoterule\relax
 {\let \footnote \thanks
 \global\@topnum\z@  % Prevents figures from going at top of page.
 \null
 \vspace*{-3pt}
 \begin{center}%
  {\Large\noindent\bfseries \@title \par}%
  \vskip 14pt
  {\normalsize
   \begin{tabular}[t]{@{}c@{}}%
    \@authors
   \end{tabular}\par}%
   \vskip9pt
 \end{center}\par
 \@thanks
 %\begin{center}\vspace*{-11pt}Received:\space\@received\vspace*{-9pt}\end{center}
 }

 \setcounter{footnote}{0}%
 \global\let\thanks\relax
 \global\let\maketitle\relax
 \global\let\@thanks\@empty
 \global\let\@author\@empty
 \global\let\@date\@empty
 \global\let\@title\@empty
 \global\let\title\relax
 \global\let\author\relax
 \global\let\date\relax
 \global\let\and\relax
}
\setcounter{secnumdepth}{3}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\renewcommand\thesection   {\@arabic\c@section}
\renewcommand\thesubsection  {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph  {\thesubsubsection.\@arabic\c@paragraph}
\def\@seccntformat#1{\csname the#1\endcsname.\hskip 0.5em}
\newcommand\section{\@startsection {section}{1}{\z@}%
                   {-\baselineskip}
                   {6\p@ \@plus \z@ \@minus\p@}%
                   {\centering\normalsize\uppercase}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                   {-\baselineskip}
                   {12\p@ \@plus 3\p@ \@minus \p@}%
                   {\centering\normalfont\normalsize\itshape}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                   {12\p@ \@plus 6\p@ \@minus 3\p@}%
                   {-1\baselineskip}
                   {\normalfont\normalsize\itshape}}
\newcommand\paragraph{\@startsection{paragraph}{4}{10\p@}%
                  {12\p@ \@plus 6\p@ \@minus 3\p@}%
                  {-\baselineskip}% 6\p@ \@plus 3\p@ \@minus 1\p@}%
                  {\bfseries\normalsize}}
\setlength\leftmargini {2pc}
\leftmargin \leftmargini
\setlength\leftmarginii {1.5pc}
\setlength\leftmarginiii {1.5pc}
\setlength\leftmarginiv {1.5pc}
\setlength\leftmarginv  {1pc}
\setlength\leftmarginvi {1pc}
\setlength \labelsep {0.5em}
\setlength \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\setlength\partopsep{\z@ \@plus 1\p@}
\@beginparpenalty -\@lowpenalty
\@endparpenalty  -\@lowpenalty
\@itempenalty   -\@lowpenalty
\def\@listi{\leftmargin\leftmargini
      \parsep\z@  \@plus1\p@ \@minus1\p@
      \topsep 12\p@ \@plus1\p@ \@minus1\p@
      \itemsep\z@  \@plus1\p@ \@minus1\p@}
\let\@listI\@listi
\@listi
\def\makeRLlabel#1{\rlap{#1}\hss}
\def\makeRRlabel#1{\hss\llap{#1}}
\def\@listii {\leftmargin\leftmarginii
       \labelwidth\leftmarginii
       \advance\labelwidth-\labelsep
       \topsep  6\p@ \@plus\p@ \@minus\p@
       \parsep  \z@
       \itemsep  \parsep
       \let\makelabel\makeRRlabel}
\def\@listiii{\leftmargin\leftmarginiii
       \labelwidth\leftmarginiii
       \advance\labelwidth-\labelsep
       \topsep  6\p@ \@plus\p@\@minus\p@
       \parsep  \z@
       \partopsep \z@
       \itemsep  \topsep
       \let\makelabel\makeRRlabel}
\def\@listiv {\leftmargin\leftmarginiv
       \labelwidth\leftmarginiv
       \advance\labelwidth-\labelsep
       \let\makelabel\makeRRlabel}
\def\@listv {\leftmargin\leftmarginv
       \labelwidth\leftmarginv
       \advance\labelwidth-\labelsep
       \let\makelabel\makeRRlabel}
\def\@listvi {\leftmargin\leftmarginvi
       \labelwidth\leftmarginvi
       \advance\labelwidth-\labelsep
       \let\makelabel\makeRRlabel}
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii}
\newcommand\labelenumiv{\theenumiv}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{$\bullet$}
\newcommand\labelitemii{\normalfont\bfseries --}
\newcommand\labelitemiii{$\m@th\ast$}
\newcommand\labelitemiv{$\m@th\cdot$}
\newenvironment{description}
        {\list{}{\labelwidth\z@ \itemindent-\leftmargin
            \let\makelabel\descriptionlabel}}
        {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                 \normalfont\bfseries #1}
\newenvironment{abstract}{%
   \list{}{\leftmargin1.5pc\rightmargin\leftmargin
       \parsep\z@ \@plus\p@
       \listparindent 1em\labelsep \z@}
       \item[]\item\relax\noindent\textbf{Summary}\unskip\hspace*{10pt}\small}%
   {\ifx\@keywords\@empty
   \else
    \par\vspace*{5pt}%% really 10pt
    \noindent\textbf{Keywords}:\unskip\hspace*{10pt}{\itshape \@keywords}%
    \vspace*{-20pt}
   \fi
   \endlist\noindent
      }
\let\@keywords\@empty
\def\keyword#1{%
 \ifx\@keywords\@empty
  \def\@keywords{#1}
 \else
  \g@addto@macro\@keywords{; #1}
 \fi}
\def\keywords#1{%
 \ifx\@keywords\@empty
  \def\@keywords{#1}
 \else
  \g@addto@macro\@keywords{#1}
 \fi}
\def\keywords#1{\gdef\@keywords{#1}}

\newenvironment{verse}
        {\let\\\@centercr
        \list{}{\itemsep   \z@
            \itemindent  -2pc%
            \listparindent\itemindent
            \rightmargin \leftmargin
            \advance\leftmargin 2pc}%
        \item\relax}
        {\endlist}
\newenvironment{quotation}
        {\list{}{\listparindent 1em%
            \itemindent  \listparindent
            \leftmargin 2.5pc \rightmargin\leftmargin
            \parsep    \z@ \@plus\p@}%
        \item\relax\small}
        {\endlist}
\newenvironment{quote}
        {\list{}{\leftmargin 2.5pc\rightmargin\leftmargin}%
        \item\relax\small}
        {\endlist}
\def\@thmcountersep{.}
\def\@begintheorem#1#2{\trivlist
  \item[\hskip \labelsep{\noindent\scshape #1 #2.}]\itshape}
\def\@opargbegintheorem#1#2#3{\trivlist
   \item[\hskip \labelsep{\noindent\scshape #1\ #2. (#3)}]\itshape}
\def\newdefinition#1{%
 \@ifnextchar[{\@odef{#1}}{\@ndef{#1}}}
\def\@ndef#1#2{%
 \@ifnextchar[{\@xndef{#1}{#2}}{\@yndef{#1}{#2}}}
\def\@xndef#1#2[#3]{%
 \expandafter\@ifdefinable\csname #1\endcsname
  {\@definecounter{#1}\@newctr{#1}[#3]%
   \expandafter\xdef\csname the#1\endcsname{%
    \expandafter\noexpand\csname the#3\endcsname \@defcountersep
     \@defcounter{#1}}%
   \global\@namedef{#1}{\@prf{#1}{#2}}%
   \global\@namedef{end#1}{\@enddefinition}}}
\def\@yndef#1#2{%
 \expandafter\@ifdefinable\csname #1\endcsname
  {\@definecounter{#1}%
   \expandafter\xdef\csname the#1\endcsname{\@defcounter{#1}}%
   \global\@namedef{#1}{\@def{#1}{#2}}%
   \global\@namedef{end#1}{\@enddefinition}}}
\def\@odef#1[#2]#3{%
 \@ifundefined{c@#2}{\@nocounterr{#2}}%
  {\expandafter\@ifdefinable\csname #1\endcsname
  {\global\@namedef{the#1}{\@nameuse{the#2}}%
 \global\@namedef{#1}{\@def{#2}{#3}}%
 \global\@namedef{end#1}{\@enddefinition}}}}
\def\@def#1#2{%
 \refstepcounter{#1}%
 \@ifnextchar[{\@ydef{#1}{#2}}{\@xdef{#1}{#2}}}
\def\@xdef#1#2{%
 \@begindefinition{#2}{\csname the#1\endcsname}\ignorespaces}
\def\@ydef#1#2[#3]{%
 \@opargbegindefinition{#2}{\csname the#1\endcsname}{#3}\ignorespaces}
\def\@defcounter#1{\noexpand\arabic{#1}}
\def\@defcountersep{.}
\def\@begindefinition#1#2{\trivlist
  \item[\hskip \labelsep{\noindent\scshape #1 #2.}]\rmfamily}
\def\@opargbegindefinition#1#2#3{\trivlist
   \item[\hskip \labelsep{\noindent\scshape #1\ #2. (#3)}]\rmfamily}
\def\@enddefinition{\endtrivlist}
\def\@begintheorem#1#2{\trivlist
  \item[\hskip \labelsep{\noindent\scshape #1\ #2.}]\itshape}
\def\@opargbegintheorem#1#2#3{\trivlist
   \item[\hskip \labelsep{\noindent\scshape #1\ #2. (#3)}]\itshape}
\def\newproof#1{%
 \@ifnextchar[{\@oprf{#1}}{\@nprf{#1}}}
\def\@nprf#1#2{%
 \@ifnextchar[{\@xnprf{#1}{#2}}{\@ynprf{#1}{#2}}}
\def\@xnprf#1#2[#3]{%
 \expandafter\@ifdefinable\csname #1\endcsname
  {\@definecounter{#1}\@newctr{#1}[#3]%
   \expandafter\xdef\csname the#1\endcsname{%
    \expandafter\noexpand\csname the#3\endcsname \@prfcountersep
     \@prfcounter{#1}}%
   \global\@namedef{#1}{\@prf{#1}{#2}}%
   \global\@namedef{end#1}{\@endproof}}}
\def\@ynprf#1#2{%
 \expandafter\@ifdefinable\csname #1\endcsname
  {\@definecounter{#1}%
   \expandafter\xdef\csname the#1\endcsname{\@prfcounter{#1}}%
   \global\@namedef{#1}{\@prf{#1}{#2}}%
   \global\@namedef{end#1}{\@endproof}}}
\def\@oprf#1[#2]#3{%
 \@ifundefined{c@#2}{\@nocounterr{#2}}%
  {\expandafter\@ifdefinable\csname #1\endcsname
  {\global\@namedef{the#1}{\@nameuse{the#2}}%
 \global\@namedef{#1}{\@prf{#2}{#3}}%
 \global\@namedef{end#1}{\@endproof}}}}
\def\@prf#1#2{%
 \refstepcounter{#1}%
 \@ifnextchar[{\@yprf{#1}{#2}}{\@xprf{#1}{#2}}}
\def\@xprf#1#2{%
 \@beginproof{#2}{\csname the#1\endcsname}\ignorespaces}
\def\@yprf#1#2[#3]{%
 \@opargbeginproof{#2}{\csname the#1\endcsname}{#3}\ignorespaces}
\def\@prfcounter#1{\noexpand\arabic{#1}}
\def\@prfcountersep{.}
\def\@beginproof#1#2{\trivlist
  \item[\hskip \labelsep{\noindent\scshape #1.}]\rmfamily}
\def\@opargbeginproof#1#2#3{\trivlist
   \item[\hskip \labelsep{\noindent\scshape #1. (#3)}]\rmfamily}
\def\@endproof{\endtrivlist}
\newcommand*{\qed}{\ifmmode\Box\else$\Box$\fi}
\newproof{proof}{Proof}
\def\@nthm#1#2{\@ifnextchar[{\@xnthm{#1}{#2}}{\@xnthm{#1}{#2}[section]}}
\def\@ndef#1#2{\@ifnextchar[{\@xndef{#1}{#2}}{\@xndef{#1}{#2}[section]}}
\newcommand\appendix{\par
 \setcounter{section}{0}%
 \setcounter{subsection}{0}%
 \renewcommand\thesection{\@Alph\c@section}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{0.4\p@}
\setlength\doublerulesep{0.4\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{0.4\p@}
%\@addtoreset{equation}{section}
\renewcommand\theequation{\thesection.\@arabic\c@equation}
\chardef\@ne=1
\chardef\tw@=2
\chardef\thr@@=3
\def\hb@xt@{\hbox to}
\def\eqnarray{%
  \stepcounter{equation}%
  \def\@currentlabel{\p@equation\theequation}%
  \global\@eqnswtrue
  \m@th
  \global\@eqcnt\z@
  \tabskip\@centering
  \let\\\@eqncr
  $$\everycr{}\halign to\displaywidth\bgroup
    \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
   &\global\@eqcnt\@ne\hskip 0.5\arraycolsep \hfil${##}$\hfil
   &\global\@eqcnt\tw@ \hskip 0.5\arraycolsep
     $\displaystyle{##}$\hfil\tabskip\@centering
   &\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
     \tabskip\z@skip
   \cr
}
\def\endeqnarray{%
   \@@eqncr
   \egroup
   \global\advance\c@equation\m@ne
  $$\global\@ignoretrue
}
\newcounter{eqnparts}
\setcounter{eqnparts}{0}
\def\parts{\refstepcounter{equation}%
      \setcounter{eqnparts}{\value{equation}}%
      \setcounter{equation}{0}%
      \def\theequation{\arabic{eqnparts}{\itshape\alph{equation}}}}
\def\endparts{\def\theequation{\arabic{equation}}%
       \setcounter{equation}{\value{eqnparts}}}
\newcounter{figure}
\renewcommand\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\textbf{\footnotesize\figurename~\thefigure}}
\newenvironment{figure}
        {\@float{figure}}
        {\end@float}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\textbf{\footnotesize\tablename~\thetable}}
\newenvironment{table}
        {\@float{table}}
        {\end@float}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{6\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
 \vskip\abovecaptionskip
 \sbox\@tempboxa{#1\textbf{.} #2}%
 \ifdim \wd\@tempboxa >\hsize
  #1\textbf{.} #2\par
 \else
  \global \@minipagefalse
  \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
 \fi
 \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{1}

\newenvironment{thebibliography}[1]
   {\section*{\refname
    \@mkboth{\refname}{\refname}}%
   \list{\@biblabel{\@arabic\c@enumiv}}%
      {\settowidth\labelwidth{\@biblabel{[#1]}}%
      \leftmargin\z@
      \advance\leftmargin\labelsep\labelwidth\z@%
      \leftmargin 1.5pc
      \itemindent-\leftmargin}
      \footnotesize
      \parindent\z@
      \parskip\z@ plus .1pt\relax
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax}
   {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
   \endlist}
\newcommand\newblock{\hskip0.11em\@plus0.33em\@minus0.07em}
\let\@openbib@code\@empty
\renewcommand*{\@biblabel}[1]{}
\def\@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
 \let\@citea\@empty
 \@cite{\@for\@citeb:=#2\do
  {\@citea\def\@citea{,\penalty\@m\ }%
   \def\@tempa##1##2\@nil{\edef\@citeb{\if##1\space##2\else##1##2\fi}}%
   \expandafter\@tempa\@citeb\@nil
%%%%   \hbox{\bf{\csname b@\@citeb\endcsname}}}}{#1}}
   \hbox{\link{\@citeb}{\csname b@\@citeb\endcsname}}}}{#1}}
%%%%\def\@cite#1#2{({#1\if@tempswa , #2\fi})}
\def\@cite#1#2{[{#1\if@tempswa , #2\fi}]}
\newcounter{pdfrefcount}
\setcounter{pdfrefcount}{1}
\def\@lbibitem[#1]#2{\item[%
            \dest{#2}{\hbox to 0pt{%
             {\special{color push}%
             \special{color rgb 1.0 1.0 1.0}|%
             \special{color pop}}\hss%
            }}%
            \dest{\arabic{pdfrefcount}}{\hbox to 0pt{%
             {\special{color push}%
             \special{color rgb 1.0 1.0 1.0}|%
             \special{color pop}}\hss%
            }}%
           \addtocounter{pdfrefcount}{1}%
 \@biblabel{#1}\hfill]\if@filesw
   {\let\protect\noexpand
    \immediate
    \write\@auxout{\string\bibcite{#2}{#1}}}\fi\ignorespaces}
\def\@bibitem#1{\item\dest{#1}{\hbox to 0pt{%
             {\special{color push}%
             \special{color rgb 1.0 1.0 1.0}|%
             \special{color pop}}\hss%
            }}%
            \dest{\arabic{pdfrefcount}}{\hbox to 0pt{%
             {\special{color push}%
             \special{color rgb 1.0 1.0 1.0}|%
             \special{color pop}}\hss%
            }}%
           \addtocounter{pdfrefcount}{1}%
    \if@filesw \immediate\write\@auxout
    {\string\bibcite{#1}{?}}\fi\ignorespaces}
%\def\label#1{\dest{#1}{}\@bsphack
% \protected@write\@auxout{}%
%     {\string\newlabel{#1}{{\apdflink{#1}{{\strut\@currentlabel}}}{\apdflink{#1}{\strut\thepage}}}}%
% \@esphack}
\renewcommand\footnoterule{%
 \kern-3\p@
 \hrule width 40\columnwidth height \z@
 \kern3\p@}
%\@addtoreset{footnote}{page}
\def\@fnsymbol#1{\ensuremath{\ifcase#1\or \dagger\or \ddagger\or
  \mathsection\or \mathparagraph\or \|\or *\or \dagger\dagger
  \or \ddagger\ddagger\or ** \else\@ctrerr\fi}}
\def\thefootnote{\@fnsymbol\c@footnote}
\renewcommand{\thefootnote}{\arabic{footnote}}

\newcommand\@makefntext[1]{%
 \noindent
 \hb@xt@0.725pc{%
  \hss\llap{\@textsuperscript{\@thefnmark}}}#1}%
\newcommand\refname{REFERENCES}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\newcommand\today{}
\edef\today{\ifcase\month\or
 January\or February\or March\or April\or May\or June\or
 July\or August\or September\or October\or November\or December\fi
 \space\number\day, \number\year}
\pagestyle{ectj}
\pagenumbering{arabic}
\AtBeginDocument{\NAT@numbersfalse}
\frenchspacing
\RequirePackage[authoryear]{natbib}

\AtBeginDocument{\maketitle}
\AtEndDocument{\par\clearpage\label@lastpage}
{\catcode`\:=12 \catcode`\-=12 \catcode`\>=12 \catcode`\<=12 %
 \gdef\codeof#1{\expandafter\codeof@\meaning#1<-:}%
 \gdef\codeof@#1:->#2<-:{#2}}

\overfullrule0pt
\let\citep\cite
\let\citeN\citet
\let\citeANP\citeauthor
\let\citeyearNP\citeyear
