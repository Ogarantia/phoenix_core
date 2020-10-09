# renders a header file defining replacement macros for every symbol wrapped with HIDENAME macro

echo "// This file is generated automatically. Do not edit it. Be nice. Please."
echo ""
echo "#ifdef UPSTRIDE_DEBUG"
echo "#define HIDENAME(_) _"
echo "#else"
echo "#define HIDENAME(_) __##_"

grep --include '*.cu' -Ero 'HIDENAME\([A-Za-z0-9_]+\)' . |\
    sort -u |\
    awk  -F ':'\
        '{printf "#define __" substr($2,10,length($2)-10) " \\\n     __symbol%03d    // " $1 "\n", NR}'

echo "#endif"