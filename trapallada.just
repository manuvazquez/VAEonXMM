import '~/scripts/common.just'

home_dir := env_var('HOME')

# root_dir := join(home_dir,'trapallada')

ls:
  just -f "{{source_file()}}" --list

[default]
local: (cd-vscode-guess join(home_dir,'trapallada'))

remote: (remote-vscode-guess 'trapallada')
