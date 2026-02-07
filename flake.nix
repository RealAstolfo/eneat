{
  description = "ENEAT - NeuroEvolution of Augmenting Topologies with C++20 Coroutines";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    exstd = {
      url = "github:RealAstolfo/exstd";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    ethreads = {
      url = "github:RealAstolfo/ethreads";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.exstd.follows = "exstd";
    };
    emath = {
      url = "github:RealAstolfo/emath";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, exstd, ethreads, emath }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        exstdPkg = exstd.packages.${system}.default;
        ethreadsPkg = ethreads.packages.${system}.default;
        emathPkg = emath.packages.${system}.default;
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "eneat";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [ zig gnumake pkg-config ];
          buildInputs = with pkgs; [ zlib raylib ];

          postUnpack = ''
            rm -rf $sourceRoot/vendors
            mkdir -p $sourceRoot/vendors
            cp -r ${exstdPkg.passthru.src-with-vendors} $sourceRoot/vendors/exstd
            cp -r ${ethreadsPkg.passthru.src-with-vendors} $sourceRoot/vendors/ethreads
            cp -r ${emathPkg.passthru.src-with-vendors} $sourceRoot/vendors/emath
            chmod -R u+w $sourceRoot/vendors
          '';

          buildPhase = ''
            make ai.o
          '';

          installPhase = ''
            mkdir -p $out/include $out/lib
            cp -r include/* $out/include/
            cp ai.o $out/lib/
          '';

          passthru.src-with-vendors = pkgs.runCommand "eneat-src" {} ''
            cp -r ${self} $out
            chmod -R u+w $out
            rm -rf $out/vendors
            mkdir -p $out/vendors
            cp -r ${exstdPkg.passthru.src-with-vendors} $out/vendors/exstd
            cp -r ${ethreadsPkg.passthru.src-with-vendors} $out/vendors/ethreads
            cp -r ${emathPkg.passthru.src-with-vendors} $out/vendors/emath
            chmod -R u+w $out/vendors
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            zig
            gcc
            gnumake
            zlib
            pkg-config
            raylib
            valgrind
            d2
          ];

          shellHook = ''
            mkdir -p vendors
            if [ ! -d vendors/exstd ] || [ -L vendors/exstd ]; then
              rm -rf vendors/exstd
              cp -r ${exstdPkg.passthru.src-with-vendors} vendors/exstd
              chmod -R u+w vendors/exstd
            fi
            if [ ! -d vendors/ethreads ] || [ -L vendors/ethreads ]; then
              rm -rf vendors/ethreads
              cp -r ${ethreadsPkg.passthru.src-with-vendors} vendors/ethreads
              chmod -R u+w vendors/ethreads
            fi
            if [ ! -d vendors/emath ] || [ -L vendors/emath ]; then
              rm -rf vendors/emath
              cp -r ${emathPkg.passthru.src-with-vendors} vendors/emath
              chmod -R u+w vendors/emath
            fi
            echo "ENEAT development environment"
            echo "  Build: make all"
            echo "  Run:   ./neat_coro"
            echo "  Docs:  nix build .#docs"
          '';
        };

        # Profiling shell with perf and flamegraph tools
        devShells.profile = pkgs.mkShell {
          buildInputs = with pkgs; [
            zig
            gcc
            gnumake
            zlib
            pkg-config
            raylib
            valgrind
            linuxPackages.perf
            flamegraph
            hotspot
          ];

          shellHook = ''
            mkdir -p vendors
            if [ ! -d vendors/exstd ] || [ -L vendors/exstd ]; then
              rm -rf vendors/exstd
              cp -r ${exstdPkg.passthru.src-with-vendors} vendors/exstd
              chmod -R u+w vendors/exstd
            fi
            if [ ! -d vendors/ethreads ] || [ -L vendors/ethreads ]; then
              rm -rf vendors/ethreads
              cp -r ${ethreadsPkg.passthru.src-with-vendors} vendors/ethreads
              chmod -R u+w vendors/ethreads
            fi
            if [ ! -d vendors/emath ] || [ -L vendors/emath ]; then
              rm -rf vendors/emath
              cp -r ${emathPkg.passthru.src-with-vendors} vendors/emath
              chmod -R u+w vendors/emath
            fi
            echo "ENEAT profiling environment"
            echo "  Profile: make profile"
            echo "  Hotspot: hotspot perf.data (per-thread analysis)"
            echo "  Output:  profile.svg (flamegraph), profile.txt (text summary)"
          '';
        };

        # Documentation package - builds D2 diagrams to SVG
        packages.docs = pkgs.stdenv.mkDerivation {
          pname = "eneat-docs";
          version = "0.1.0";
          src = ./docs;

          nativeBuildInputs = with pkgs; [ d2 ];

          buildPhase = ''
            mkdir -p out

            # Build all D2 diagrams to SVG
            for f in *.d2; do
              if [ "$f" != "theme.d2" ]; then
                name="''${f%.d2}"
                echo "Building $name.svg..."
                d2 --theme 200 "$f" "out/$name.svg"
              fi
            done
          '';

          installPhase = ''
            mkdir -p $out
            cp out/*.svg $out/
            cp *.d2 $out/
          '';

          meta = with pkgs.lib; {
            description = "ENEAT architecture documentation diagrams";
            license = licenses.mit;
          };
        };
      }
    );
}
