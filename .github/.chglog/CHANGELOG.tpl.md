# What's New

{{ if .Versions -}}

<a name="unreleased"></a>
## [Unreleased]

{{ if .Unreleased.CommitGroups -}}
{{ range .Unreleased.CommitGroups -}}
### {{ .Title }}
{{ range .Commits -}}
- {{ if .Scope }}**{{ .Scope }}:** {{ end }}{{ .Subject }}
{{ end }}
{{ end -}}
{{ end -}}
{{ end -}}

{{ range .Versions }}
{{- if not (eq .Tag.Name "0.6.0" "0.6.1" "0.6.2" "0.6.3" "0.6.4" "0.6.5" "0.6.6") }}
<a name="{{ .Tag.Name }}"></a>
## {{ if .Tag.Previous }}[{{ .Tag.Name }}]{{ else }}{{ .Tag.Name }}{{ end }} - {{ datetime "2006-02-01" .Tag.Date }}
{{ range .CommitGroups -}}
### {{ .Title }}
{{ range .Commits -}}
- {{ if .Scope }}**{{ .Scope }}:** {{ end }}{{ .Subject }}
{{ end }}
{{ end -}}

{{- if .RevertCommits -}}
### Reverts
{{ range .RevertCommits -}}
- {{ .Revert.Header }}
{{ end }}
{{ end -}}


{{- if .NoteGroups -}}
{{ range .NoteGroups -}}
### {{ .Title }}
{{ range .Notes }}
{{ .Body }}
{{ end }}
{{ end -}}
{{ end -}}
{{ end -}}

{{ end -}}



<a name="0.6.6"></a>
## [0.6.6] - 2022-02-12

### Improved/Fixed

- improved build process
- improved periodic neighbor computation
- Improved documentation
- Improved CI/Tests
- minor fixes


<a name="0.6.5"></a>
## [0.6.5] - 2022-02-03

### New Features

- 2D/3D ``Shape`` object for easier lattice construction.
- repeat/extend built lattices.

### Improved/Fixed

- improved build process
- improved periodic neighbor computation (still not stable)
- added/improved tests
- improved plotting
- added more docstrings
- fixed multiple bugs

## 0.6.4 - 2022-01-27


{{- if .Versions }}

[Unreleased]: {{ .Info.RepositoryURL }}/compare/{{ $latest := index .Versions 0 }}{{ $latest.Tag.Name }}...HEAD
{{ range .Versions -}}
{{ if .Tag.Previous -}}
[{{ .Tag.Name }}]: {{ $.Info.RepositoryURL }}/compare/{{ .Tag.Previous.Name }}...{{ .Tag.Name }}
{{ end -}}
{{ end -}}
{{ end -}}
