import { fireEvent, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  example,
  paragraph,
  READY,
  renderEvaluation,
  setupEvaluationTests,
} from "@/test/evaluationTestUtils";

setupEvaluationTests();

/**
 * P3-19: visor navegable de ejemplos (aciertos primero, luego errores) con
 * "Anterior" y "Siguiente". Se mockea `fetch` (nunca se toca ml-api real).
 * Los datos son fixtures: el componente no tiene ninguna cifra escrita a mano.
 */

const VIEWER_NAME = "Explorador de ejemplos";

const REPORT = {
  ...READY,
  successes: [
    example("000008_000001", "cat", "cat", 0.9534333348274231),
    example("000014_000013", "dog", "dog", 0.8812),
    example("000020_000021", "cat", "cat", 0.7203),
  ],
  errors: [
    example("000275_000320", "dog", "cat", 0.5771472454071045),
    example("000301_000302", "cat", "dog", 0.6123),
  ],
};

type ViewerState = Readonly<{
  kind: "Acierto" | "Error";
  position: string;
  cropId: string;
  real: string;
  pred: string;
  canGoBack: boolean;
  canGoForward: boolean;
}>;

async function openViewer() {
  return within(await screen.findByRole("region", { name: VIEWER_NAME }));
}

async function renderViewer(payload: unknown = REPORT) {
  renderEvaluation(payload);
  await openViewer();
}

function click(name: string, times = 1) {
  for (let step = 0; step < times; step += 1) {
    fireEvent.click(screen.getByRole("button", { name }));
  }
}

function expectButton(name: string, enabled: boolean) {
  const button = screen.getByRole("button", { name });
  if (enabled) expect(button).toBeEnabled();
  else expect(button).toBeDisabled();
}

/** Comprueba todo lo que muestra el visor en un momento dado. */
async function expectViewerAt(state: ViewerState) {
  const viewer = await openViewer();
  expect(viewer.getByText(state.kind)).toBeInTheDocument();
  expect(viewer.getByText(state.position)).toBeInTheDocument();
  expect(viewer.getByAltText(`Recorte seleccionado ${state.cropId}`)).toHaveAttribute(
    "src",
    `/ml-api/crops/${state.cropId}`
  );
  expect(viewer.getByText(paragraph(state.real))).toBeInTheDocument();
  expect(viewer.getByText(paragraph(state.pred))).toBeInTheDocument();
  expectButton("Anterior", state.canGoBack);
  expectButton("Siguiente", state.canGoForward);
}

const FIRST_SUCCESS: ViewerState = {
  kind: "Acierto",
  position: "1 de 5",
  cropId: "000008_000001",
  real: "real cat",
  pred: "pred cat · 95.34%",
  canGoBack: false,
  canGoForward: true,
};

const SECOND_SUCCESS: ViewerState = {
  kind: "Acierto",
  position: "2 de 5",
  cropId: "000014_000013",
  real: "real dog",
  pred: "pred dog · 88.12%",
  canGoBack: true,
  canGoForward: true,
};

const FIRST_ERROR: ViewerState = {
  kind: "Error",
  position: "4 de 5",
  cropId: "000275_000320",
  real: "real dog",
  pred: "pred cat · 57.71%",
  canGoBack: true,
  canGoForward: true,
};

const LAST_ERROR: ViewerState = {
  kind: "Error",
  position: "5 de 5",
  cropId: "000301_000302",
  real: "real cat",
  pred: "pred dog · 61.23%",
  canGoBack: true,
  canGoForward: false,
};

describe("EvaluationPage: navegación de la galería (P3-19)", () => {
  it("abre en el primer acierto con su recorte, clases y probabilidad", async () => {
    await renderViewer();

    await expectViewerAt(FIRST_SUCCESS);
  });

  it("Siguiente avanza al ejemplo siguiente y actualiza posición, recorte y datos", async () => {
    await renderViewer();
    click("Siguiente");

    await expectViewerAt(SECOND_SUCCESS);
  });

  it("al pasar del último acierto al primer error cambia la etiqueta a Error", async () => {
    await renderViewer();
    click("Siguiente", 2);
    expect((await openViewer()).getByText("Acierto")).toBeInTheDocument();
    click("Siguiente");

    await expectViewerAt(FIRST_ERROR);
  });

  it("Anterior regresa al ejemplo previo", async () => {
    await renderViewer();
    click("Siguiente", 2);
    click("Anterior");

    await expectViewerAt(SECOND_SUCCESS);
  });

  it("en el último ejemplo Siguiente queda deshabilitado y no da la vuelta", async () => {
    await renderViewer();
    click("Siguiente", 4);

    await expectViewerAt(LAST_ERROR);
  });

  it("sin aciertos navega solo por los errores", async () => {
    await renderViewer({ ...REPORT, successes: [] });

    await expectViewerAt({
      ...FIRST_ERROR,
      position: "1 de 2",
      canGoBack: false,
    });
  });

  it("sin aciertos ni errores no muestra el visor", async () => {
    renderEvaluation({ ...REPORT, successes: [], errors: [] });

    await screen.findByText("98.63%");
    expect(screen.queryByRole("region", { name: VIEWER_NAME })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Siguiente" })).not.toBeInTheDocument();
  });
});
