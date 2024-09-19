fn main() {
    //println!("cargo:rustc-link-search=native=.");
    //println!("cargo:rustc-link-lib=static=yourlibname"); // If using a static library
    //println!("cargo:rustc-link-lib=dylib=yourlibname");  // If using a dynamic library

    // If you have individual .o files
    println!("cargo:rustc-link-arg=Trainer.o");
    println!("cargo:rustc-link-arg=Compressor.o");
}

