path_req_txt = "requirements_orig.txt"

pkgs_new = []
with open(path_req_txt) as myFile:
    pkgs = myFile.read()
    pkgs = pkgs.splitlines()
    for idx, pkg in enumerate(pkgs):
        if idx > 2:
            pkg_mod = pkg.split("=")[0:2]
            new_pkg = f"{pkg_mod[0]}=={pkg_mod[1]}"
            pkgs_new.append(new_pkg)

file = open("requirements.txt", "w")
for pkg_new in pkgs_new:
    file.write(pkg_new + "\n")
file.close()
