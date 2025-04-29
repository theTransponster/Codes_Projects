import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

public class PanelResultados extends JPanel{
    private JButton btExportar = new JButton("Expotar a HTML"),
                    btCopiarPortapapeles = new JButton("Copiar al Portapapeles");
    //INGRESOS
    JTextField Nombre = new JTextField(12);
    JTextField RFC = new JTextField(12);
    JTextField SaldoMensual = new JTextField(12);
    JTextField IngresoAnual = new JTextField(12);
    JTextField Aguinaldo = new JTextField(12);
    JTextField AguinaldoExcento = new JTextField(12);
    JTextField AguinaldoGravado = new JTextField(12);
    JTextField PrimaVacacional = new JTextField(12);
    JTextField PrimaVacacionalExcento = new JTextField(12);
    JTextField PrimaVacacionalGravado = new JTextField(12);
    JTextField TotalIngresosGravados = new JTextField(12);

    //DEDUCCIONES
    JTextField MedicinaHospitalarios = new JTextField(8);
    JTextField GastosFunerarios = new JTextField(8);
    JTextField SGMM = new JTextField(8);
    JTextField Hipotecario = new JTextField(8);
    JTextField Donativos = new JTextField(8);
    JTextField SubcuentaRetiro = new JTextField(8);
    JTextField TransporteEscolar = new JTextField(8);
    JTextField NivelEducativo = new JTextField(8);
    JTextField MaximoDeduccionColegiatura = new JTextField(8);
    JTextField ColegiaturaPagada = new JTextField(8);
    JTextField TotalDeduccion = new JTextField(8);

    //BALANCE
    JTextField DeduccionesPermitidas = new JTextField(12);
    JTextField MontoCalculadoISR = new JTextField(12);
    JTextField CuotaFija = new JTextField(12);
    JTextField PorcentajeExcedido = new JTextField(12);
    JTextField PagoExcedido = new JTextField(12);
    JTextField TotalPagar = new JTextField(12);

    JFileChooser fc;
    
    private PanelControles pcc;
    private Deduccion d1;
    private Persona per;
    private PanelControles parent;
    JPanel panelRightLeft;
    
    public PanelResultados(){
        super();
        this.fc=new JFileChooser();
        //PANELES
        JPanel panelTop = new JPanel(new BorderLayout());
        JPanel panelLeft = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JPanel panelRight = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        this.setPreferredSize(new Dimension(900,700));
        this.setBackground(SystemColor.LIGHT_GRAY);

        //PANEL TOP
        panelTop.setBackground(SystemColor.LIGHT_GRAY);
        GridLayout gl1 = new GridLayout(3,1,10,10);
        panelTop.setLayout(gl1);
        panelTop.add(new JLabel(" "));
        panelTop.add(new JLabel(" **********************************************************************   RESULTADOS   ***********************************************************************"));
        panelTop.add(new JLabel(" "));
        this.add(panelTop,BorderLayout.NORTH);

        //PANEL LEFT
        panelLeft.setBackground(SystemColor.LIGHT_GRAY);
        panelLeft.setPreferredSize(new Dimension(350,700));
        //panelLeft.setLayout(new BoxLayout(panelLeft, BoxLayout.Y_AXIS));
        panelLeft.add(new JLabel("*************************** INGRESOS ***************************"));
        JPanel panelLeftLeft = new JPanel(new BorderLayout());
        panelLeftLeft.setBackground(SystemColor.LIGHT_GRAY);
        JPanel panelLeftRight = new JPanel(new BorderLayout());
        panelLeftRight.setBackground(SystemColor.LIGHT_GRAY);
        GridLayout glTextLabels = new GridLayout(12,12,10,10); // TextLabels
        panelLeftLeft.setLayout(glTextLabels);
        GridLayout glTextFields = new GridLayout(12,10,10,6); // TextFields
        panelLeftRight.setLayout(glTextFields);
        panelLeftLeft.add(new JLabel("Nombre:   ", SwingConstants.RIGHT));
        panelLeftRight.add(Nombre);
        //Nombre.setText("Ki Won Kim");
        Nombre.setEditable(false);  //******Nombre.setText("Hola");
        panelLeftLeft.add(new JLabel("RFC:   ", SwingConstants.RIGHT));
        panelLeftRight.add(RFC);
        RFC.setEditable(false);
        panelLeftLeft.add(new JLabel("Saldo Mensual:   ", SwingConstants.RIGHT));
        panelLeftRight.add(SaldoMensual);
        SaldoMensual.setEditable(false);
        panelLeftLeft.add(new JLabel("Ingreso Anual:   ", SwingConstants.RIGHT));
        panelLeftRight.add(IngresoAnual);
        IngresoAnual.setEditable(false);
        panelLeftLeft.add(new JLabel("Aguinaldo:   ", SwingConstants.RIGHT));
        panelLeftRight.add(Aguinaldo);
        Aguinaldo.setEditable(false);
        panelLeftLeft.add(new JLabel("Aguinaldo Excento:   ", SwingConstants.RIGHT));
        panelLeftRight.add(AguinaldoExcento);
        AguinaldoExcento.setEditable(false);
        panelLeftLeft.add(new JLabel("Aguinaldo Gravado:  ", SwingConstants.RIGHT));
        panelLeftRight.add(AguinaldoGravado);
        AguinaldoGravado.setEditable(false);
        panelLeftLeft.add(new JLabel("Prima Vacacional:  ", SwingConstants.RIGHT));
        panelLeftRight.add(PrimaVacacional);
        PrimaVacacional.setEditable(false);
        panelLeftLeft.add(new JLabel("Prima Vacacional Excento:  ", SwingConstants.RIGHT));
        panelLeftRight.add(PrimaVacacionalExcento);
        PrimaVacacionalExcento.setEditable(false);
        panelLeftLeft.add(new JLabel("Prima Vacacional Gravada:   ", SwingConstants.RIGHT));
        panelLeftRight.add(PrimaVacacionalGravado);
        PrimaVacacionalGravado.setEditable(false);
        panelLeftLeft.add(new JLabel("Total Ingresos Gravado:   ", SwingConstants.RIGHT));
        panelLeftRight.add(TotalIngresosGravados);
        TotalIngresosGravados.setEditable(false);
        panelLeftLeft.add(new JLabel(" "));
        panelLeft.add(panelLeftLeft,BorderLayout.WEST);
        panelLeft.add(panelLeftRight,BorderLayout.EAST);

        //PANEL LEFT BOTTOM
        panelLeft.add(new JLabel(" "));
        panelLeft.add(new JLabel("**************************** BALANCE ****************************"));
        JPanel panelBotLeft = new JPanel(new BorderLayout());
        panelBotLeft.setBackground(SystemColor.LIGHT_GRAY);
        JPanel panelBotRight = new JPanel(new BorderLayout());
        panelBotRight.setBackground(SystemColor.LIGHT_GRAY);
        GridLayout glTextLabels2 = new GridLayout(8,12,10,10); // TextLabels
        panelBotLeft.setLayout(glTextLabels2);
        GridLayout glTextFields2 = new GridLayout(8,10,10,10); // TextFields
        panelBotRight.setLayout(glTextFields2);
        panelBotLeft.add(new JLabel("Deducciones Permitidas:   ", SwingConstants.RIGHT));
        panelBotRight.add(DeduccionesPermitidas);
        DeduccionesPermitidas.setEditable(false);
        panelBotLeft.add(new JLabel("Monto Calculado de ISR:   ", SwingConstants.RIGHT));
        panelBotRight.add(MontoCalculadoISR);
        MontoCalculadoISR.setEditable(false);
        panelBotLeft.add(new JLabel("Cuota Fija:   ", SwingConstants.RIGHT));
        panelBotRight.add(CuotaFija);
        CuotaFija.setEditable(false);
        panelBotLeft.add(new JLabel("Porcentaje Excedido:   ", SwingConstants.RIGHT));
        panelBotRight.add(PorcentajeExcedido);
        PorcentajeExcedido.setEditable(false);
        panelBotLeft.add(new JLabel("Pago Excedido:   ", SwingConstants.RIGHT));
        panelBotRight.add(PagoExcedido);
        PagoExcedido.setEditable(false);
        JLabel totalPagarTitulo = new JLabel("Total a Pagar: ", SwingConstants.RIGHT);
        Font auxFont=totalPagarTitulo.getFont();
        totalPagarTitulo.setFont(new Font(auxFont.getFontName(), auxFont.getStyle(), 15));
        panelBotLeft.add(totalPagarTitulo);
        panelBotRight.add(TotalPagar);
        TotalPagar.setEditable(false);
        panelLeft.add(panelBotLeft,BorderLayout.WEST);
        panelLeft.add(panelBotRight,BorderLayout.EAST);
        PanelResultados.this.add(panelLeft,BorderLayout.WEST);

        //PANEL RIGHT
        panelRight.setBackground(SystemColor.LIGHT_GRAY);
        panelRight.setPreferredSize(new Dimension(360,700));
        panelRight.add(new JLabel("************************* DEDUCCIONES *************************"));
        this.panelRightLeft = new JPanel(new BorderLayout());
        panelRightLeft.setBackground(SystemColor.LIGHT_GRAY);
        JPanel panelRightRight = new JPanel(new BorderLayout());
        panelRightRight.setBackground(SystemColor.LIGHT_GRAY);
        GridLayout glTextLabels3 = new GridLayout(13,10,10,6); // TextLabels
        panelRightLeft.setLayout(glTextLabels3);
        GridLayout glTextFields3 = new GridLayout(13,10,10,6); // TextFields
        panelRightRight.setLayout(glTextFields3);
        panelRightLeft.add(new JLabel("Medicos y Hospitalarios:", SwingConstants.RIGHT));
        panelRightRight.add(MedicinaHospitalarios);
        MedicinaHospitalarios.setEditable(false);
        panelRightLeft.add(new JLabel("Gastos Funerarios:", SwingConstants.RIGHT));
        panelRightRight.add(GastosFunerarios);
        GastosFunerarios.setEditable(false);
        panelRightLeft.add(new JLabel("SGMM:", SwingConstants.RIGHT));
        panelRightRight.add(SGMM);
        SGMM.setEditable(false);
        panelRightLeft.add(new JLabel("Hipotecario:", SwingConstants.RIGHT));
        panelRightRight.add(Hipotecario);
        Hipotecario.setEditable(false);
        panelRightLeft.add(new JLabel("Donativos:", SwingConstants.RIGHT));
        panelRightRight.add(Donativos);
        Donativos.setEditable(false);
        panelRightLeft.add(new JLabel("Subcuentas y retiro:", SwingConstants.RIGHT));
        panelRightRight.add(SubcuentaRetiro);
        SubcuentaRetiro.setEditable(false);
        panelRightLeft.add(new JLabel("Transporte escolar:", SwingConstants.RIGHT));
        panelRightRight.add(TransporteEscolar);
        TransporteEscolar.setEditable(false);
        panelRightLeft.add(new JLabel("Nivel Educativo:", SwingConstants.RIGHT));
        panelRightRight.add(NivelEducativo);
        NivelEducativo.setEditable(false);
        panelRightLeft.add(new JLabel("Maximo deduccion colegiatura:", SwingConstants.RIGHT));
        panelRightRight.add(MaximoDeduccionColegiatura);
        MaximoDeduccionColegiatura.setEditable(false);
        panelRightLeft.add(new JLabel("Colegiatura pagada:", SwingConstants.RIGHT));
        panelRightRight.add(ColegiaturaPagada);
        ColegiaturaPagada.setEditable(false);
        panelRightLeft.add(new JLabel("Total Deduccion:", SwingConstants.RIGHT));
        panelRightRight.add(TotalDeduccion);
        TotalDeduccion.setEditable(false);
        panelRightLeft.add(new JLabel(" "));
        panelRightRight.add(new JLabel(" "));
        panelRight.add(panelRightLeft,BorderLayout.WEST);
        panelRight.add(panelRightRight,BorderLayout.EAST);
        PanelResultados.this.add(panelRight, BorderLayout.EAST);
        
       
        btCopiarPortapapeles.addActionListener(new ActionListener(){
            @Override
            public void actionPerformed(ActionEvent e) {
                copiarPortapapeles();
            }
        });
        panelRightRight.add(btCopiarPortapapeles);

    }
    public void copiarPortapapeles(){
        String str = "";
        str += "****INGRESOS****" + "\n";
        str += "Nombre: "+ Nombre.getText() + "\n";
        str += "RFC: "+ RFC.getText() + "\n";
        str += "Saldo Mensual: " + SaldoMensual.getText() + "\n";
        str += "Ingreso Anual: " + IngresoAnual.getText() + "\n";
        str += "Aguinaldo: " + IngresoAnual.getText() + "\n";
        str += "Aguinaldo Excento: " + AguinaldoExcento.getText() + "\n";
        str += "Aguinaldo Gravado: " + AguinaldoGravado.getText() + "\n";
        str += "Prima Vacacional" + PrimaVacacional.getToolTipText() + "\n";
        str += "Prima Vacacional Excento: " + PrimaVacacionalExcento.getText() + "\n";
        str += "Prima Vacacional Gravado: " + PrimaVacacionalGravado.getText() + "\n\n";

        str += "****DEDUCCIONES****" + "\n";
        str += "Medicina y HospitalizaciÃ³n: " + MedicinaHospitalarios.getText() + "\n";
        str += "Gastos Funerarios: " + GastosFunerarios.getText() + "\n";
        str += "Seguro de Gastos Médicos Mayores: " + SGMM.getText() +"\n";
        str += "Hipotecario: " + Hipotecario.getText() +"\n";
        str += "Donativos: " + Donativos.getText() + "\n";
        str += "Subcuenta y Retiro: " + SubcuentaRetiro.getText() + "\n";
        str += "Transporte Escolar: " + TransporteEscolar.getText() + "\n";
        str += "Nivel Educativo: " + NivelEducativo.getText() + "\n";
        str += "Máxima Deducción de Colegiatura: " + MaximoDeduccionColegiatura.getText() + "\n";
        str += "Colegiatura Pagada: " + ColegiaturaPagada.getText() + "\n";
        str += "Total de Deducción: " + TotalDeduccion.getText() + "\n\n";

        str += "****BALANCE****" +"\n";
        str += "Deducciones Permitidas: " + DeduccionesPermitidas.getText() + "\n";
        str += "Monto Calculado de ISR: " + MontoCalculadoISR.getText() + "\n";
        str += "Cuota Fija: " + CuotaFija.getText() + "\n";
        str += "Porcentaje Excedido" + PorcentajeExcedido.getText() + "\n";
        str += "Pago Excedido: " + PagoExcedido.getText() + "\n";
        str += "TOTAL A PAGAR: " + TotalPagar.getText() + "\n";

        StringSelection stringSelection = new StringSelection(str);
        Clipboard clpbrd = Toolkit.getDefaultToolkit().getSystemClipboard();
        clpbrd.setContents(stringSelection, null);
    }

}
