{
  description = "ENEAT - NeuroEvolution of Augmenting Topologies with C++20 Coroutines";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            zig
            gcc
            zlib
            pkg-config
            raylib
            valgrind
            d2
          ];

          shellHook = ''
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
            zlib
            pkg-config
            raylib
            valgrind
            linuxPackages.perf
            flamegraph
            hotspot  # Per-thread GUI analysis
          ];

          shellHook = ''
            echo "ENEAT profiling environment"
            echo "  Profile: make profile"
            echo "  Hotspot: hotspot perf.data (per-thread analysis)"
            echo "  Output:  profile.svg (flamegraph), profile.txt (text summary)"
          '';
        };

        packages.default = pkgs.stdenv.mkDerivation {
          pname = "eneat";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [ zig pkg-config ];
          buildInputs = with pkgs; [ zlib raylib ];

          buildPhase = ''
            make all
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp neat neat_coro $out/bin/
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
