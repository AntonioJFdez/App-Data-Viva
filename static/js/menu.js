// MENÚ RESPONSIVE: Hamburguesa y desplegables
document.addEventListener("DOMContentLoaded", function() {
    const hamburger = document.getElementById("hamburger-btn");
    const menu = document.querySelector(".navbar-menu");
    hamburger.addEventListener("click", function() {
        menu.classList.toggle("menu-open");
        hamburger.classList.toggle("is-active");
    });

    // Menú desplegable solo en móvil
    const dropdownLinks = document.querySelectorAll(".dropdown > a");
    dropdownLinks.forEach(link => {
        link.addEventListener("click", function(e) {
            if (window.innerWidth <= 900) {
                e.preventDefault();
                const submenu = this.nextElementSibling;
                submenu.classList.toggle("submenu-open");
            }
        });
    });

    // Cierra menú al hacer click fuera (opcional)
    document.addEventListener("click", function(e) {
        if (!menu.contains(e.target) && !hamburger.contains(e.target)) {
            menu.classList.remove("menu-open");
            hamburger.classList.remove("is-active");
        }
    });
});
