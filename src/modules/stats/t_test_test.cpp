
#include <cstdio>
#include <cstdlib>
#import <mach-o/dyld.h>

int main()
{
    NSObjectFileImage img; // Represents the bundle's object file
    NSModule handle;       // Handle to the loaded bundle
    NSSymbol sym;          // Represents a symbol in the bundle
    int rc;

    void (*func) (double&, double&, double&, double, double, double);
    
    rc = NSCreateObjectFileImageFromFile("/Users/qianh1/workspace/madlib/build/src/ports/postgres/9.2/lib/libmadlib.so", &img);
    if (rc != NSObjectFileImageSuccess) {
        fprintf(stderr, "Could not load bundle.\n");
        exit(-1);
    }

    handle = NSLinkModule(img, "/Users/qianh1/workspace/madlib/build/src/ports/postgres/9.2/lib/libmadlib.so", FALSE);
    sym = NSLookupSymbolInModule(handle, "updateCorrectedSumOfSquares");
    if (sym == NULL) {
        fprintf(stderr, "Could not find symbol.\n");
        exit(-2);
    }

    func = NSAddressOfSymbol(sym);

    return 0;
}
