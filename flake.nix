{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs = { self, nixpkgs, nixpkgs-python }: 
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Python setup with necessary packages
      myPythonWithPackages = pkgs.python3.withPackages (ps: with ps; [
        numpy
        matplotlib
        shapely
      ]);
    in
    {
      # Default package for `nix build .`
      defaultPackage.${system} = pkgs.mkShell {
        buildInputs = [ myPythonWithPackages ];
        shellHook = ''
          echo "kuka-tsp python shell built"
          python --version
        '';
      };

      # Development shell entry point for `nix develop`
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [ myPythonWithPackages ];
        shellHook = ''
          echo "kuka-tsp python shell activated"
          python --version
        '';
      };
    };
}
