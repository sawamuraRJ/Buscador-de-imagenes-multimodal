// Funciones compartidas por las dos páginas
function mostrarResultados(resultados) {
  const grilla = document.getElementById("resultados");
  grilla.innerHTML = "";
  if (!resultados.length) {
    grilla.innerHTML = '<p class="mensaje">No se encontraron imágenes similares.</p>';
    return;
  }
  for (const r of resultados) {
    const a = document.createElement("a");
    a.className = "tarjeta";
    a.href = r.image;
    a.target = "_blank";
    a.innerHTML = `<img src="${r.thumb}" alt="Resultado de búsqueda" loading="lazy"><span>${r.score.toFixed(2)}</span>`;
    grilla.appendChild(a);
  }
}

function mostrarMensaje(texto, esError = false) {
  const grilla = document.getElementById("resultados");
  grilla.innerHTML = `<p class="mensaje ${esError ? "error" : ""}">${texto}</p>`;
}

async function pedir(url, opciones, boton) {
  boton.disabled = true;
  mostrarMensaje("Buscando...");
  try {
    const resp = await fetch(url, opciones);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Error en la búsqueda");
    mostrarResultados(data.resultados);
  } catch (e) {
    mostrarMensaje(e.message, true);
  } finally {
    boton.disabled = false;
  }
}
