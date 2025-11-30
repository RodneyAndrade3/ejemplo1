function agregarProyecto() {
    const lista = document.getElementById("lista-proyectos");

    const nuevo = document.createElement("li");
    nuevo.textContent = "Nuevo proyecto agregado";

    lista.appendChild(nuevo);
}