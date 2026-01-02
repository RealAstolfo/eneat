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
          ];

          shellHook = ''
            echo "ENEAT development environment"
            echo "  Build: make all"
            echo "  Run:   ./neat_coro"
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
      }
    );
}
