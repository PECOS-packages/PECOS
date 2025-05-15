import re

# Read the file
with open('custom_include_paths_test.rs', 'r') as f:
    content = f.read()

# Replace parse_str_with_include_paths
pattern = r'QASMParser::parse_str_with_include_paths\(([^,]+),\s*([^)]+)\)'
replacement = r'{ let mut config = ParseConfig::default(); config.include_paths = \2.into_iter().map( < /dev/null | p| p.into()).collect(); QASMParser::parse_with_config(\1, config) }'
content = re.sub(pattern, replacement, content)

# Replace parse_str_with_include_paths_and_virtual
pattern = r'QASMParser::parse_str_with_include_paths_and_virtual\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\s*\)'
replacement = r'{ let mut config = ParseConfig::default(); config.include_paths = \2.into_iter().map(|p| p.into()).collect(); config.virtual_includes = \3.into_iter().collect(); QASMParser::parse_with_config(\1, config) }'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('custom_include_paths_test.rs', 'w') as f:
    f.write(content)
