import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.filechooser.FileNameExtensionFilter;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class PanelControles extends JPanel{
    private PanelResultados pd;
    private JButton btCalcular = new JButton("Calcular"),
                    btLimpiar = new JButton("Limpiar"),
                    btExportar=new JButton("Exportar httml");
    private JTextField Nombre = new JTextField(10);
    private JTextField RFC = new JTextField(10);
    private JTextField SaldoMensual = new JTextField(10);
    private JTextField Aguinaldo = new JTextField(10);
    private JTextField PrimaVacacional = new JTextField(10);
    private JTextField MedicinaHospitalizacion = new JTextField(10);
    private JTextField GastosFunerarios = new JTextField(10);
    private JTextField SGMM = new JTextField(10);

    private JTextField Hipotecario = new JTextField(8);
    private JTextField Donativos = new JTextField(8);
    private JTextField SubcuentaRetiro = new JTextField(8);
    private JTextField TransporteEscolar = new JTextField(8);
    private JTextField Colegiatura = new JTextField(8);
    
    private Persona persona;
    private Deduccion deduccion;
    private String[] trabajador;
    private JTextField[] campos;
    private String[] resultados;
    private Double[] pers;
    private JFileChooser fc;
    private String titulo;

    private JRadioButton rbPrimaria,
                         rbSecundaria,
                         rbPreparatoria,
                         rbNinguno;

    public PanelControles(PanelResultados pd){
        super();
        this.trabajador=new String[14];
        this.pers=new Double[14];
        this.campos=new JTextField[14];
        this.resultados=new String[28];
        this.titulo="";
        this.pd=pd;
        this.fc=new JFileChooser();
        JPanel panelLeft = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JPanel panelRight = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        this.setPreferredSize(new Dimension(550,700));  //Definir tamaÃ±o de ventana

        panelLeft.setPreferredSize(new Dimension(250,700));
        panelRight.setPreferredSize(new Dimension(250,700));

        GridLayout glLeft = new GridLayout(30,1,10,1);
        panelLeft.setLayout(glLeft);
        panelLeft.add(new JLabel("Nombre:  "));
        panelLeft.add(Nombre);
        panelLeft.add(new JLabel("RFC:  "));
        panelLeft.add(RFC);
        panelLeft.add(new JLabel("Sueldo Mensual:  "));
        panelLeft.add(SaldoMensual);
        panelLeft.add(new JLabel("Aguinaldo:  "));
        panelLeft.add(Aguinaldo);
        panelLeft.add(new JLabel("Prima Vacacional:  "));
        panelLeft.add(PrimaVacacional);
        panelLeft.add(new JLabel("Medicina y Hospitalización:  "));
        panelLeft.add(MedicinaHospitalizacion);
        panelLeft.add(new JLabel("Gastos Funerarios:  "));
        panelLeft.add(GastosFunerarios);
        panelLeft.add(new JLabel("Seguro de Gastos Médicos Mayores:  "));
        panelLeft.add(SGMM);

        GridLayout glRight = new GridLayout(30,2,10,1);
        panelRight.setLayout(glRight);
        panelRight.add(new JLabel("Hipotecario:  "));
        panelRight.add(Hipotecario);
        panelRight.add(new JLabel("Donativos:  "));
        panelRight.add(Donativos);
        panelRight.add(new JLabel("Subcuenta y Retiro:  "));
        panelRight.add(SubcuentaRetiro);
        panelRight.add(new JLabel("Transporte Escolar:  "));
        panelRight.add(TransporteEscolar);
        panelRight.add(new JLabel("Colegiatura:  "));
        panelRight.add(Colegiatura);
        panelRight.add(new JLabel(" "));

        panelRight.add(new JLabel("Nivel Escolar:  "));
        rbPrimaria = new JRadioButton("Primaria");
        rbSecundaria = new JRadioButton("Secundaria");
        rbPreparatoria = new JRadioButton("Preparatoria");
        rbNinguno = new JRadioButton("Ninguno");
        panelRight.add(rbPrimaria);
        panelRight.add(rbSecundaria);
        panelRight.add(rbPreparatoria);
        panelRight.add(rbNinguno);
        rbNinguno.setSelected(true);
        ButtonGroup bg = new ButtonGroup();
        bg.add(rbPrimaria);
        bg.add(rbSecundaria);
        bg.add(rbPreparatoria);
        bg.add(rbNinguno);
        panelRight.add(new JLabel(" "));
        //this.btCalcular.addActionListener(new ActionListener(){
        //   @Override
        //    public void actionPerformed(ActionEvent e) {
        //        System.out.println();
        //    }
        //});
        
        btCalcular.addActionListener(new ActionListener(){
        	public void actionPerformed(ActionEvent e){
        		String educacion="Ninguna";
        		String Nombre=PanelControles.this.Nombre.getText();
        		String RFC=PanelControles.this.RFC.getText();
        		if(rbPrimaria.isSelected()){
        			educacion="Primaria";
        		}else if(rbSecundaria.isSelected()){
        			educacion="Secundaria";
        		}else if(rbPreparatoria.isSelected()){
        			educacion="Preparatoria";
        		}
        		trabajador[0]=Nombre;
        		trabajador[1]=RFC;
        		trabajador[2]=educacion;
        		trabajador[3]=PanelControles.this.SaldoMensual.getText();
        		trabajador[4]=PanelControles.this.Aguinaldo.getText();
        		trabajador[5]=PanelControles.this.PrimaVacacional.getText();
        		trabajador[6]=PanelControles.this.Colegiatura.getText();
        		trabajador[7]=PanelControles.this.SGMM.getText();
        		trabajador[8]=PanelControles.this.GastosFunerarios.getText();
        		trabajador[9]=PanelControles.this.MedicinaHospitalizacion.getText();
        		trabajador[10]=PanelControles.this.Hipotecario.getText();
        		trabajador[11]=PanelControles.this.Donativos.getText();
        		trabajador[12]=PanelControles.this.SubcuentaRetiro.getText();
        		trabajador[13]=PanelControles.this.TransporteEscolar.getText();
        		
        		campos[0]=PanelControles.this.Nombre;
        		campos[1]=PanelControles.this.RFC;
        		campos[3]=PanelControles.this.SaldoMensual;
        		campos[4]=PanelControles.this.Aguinaldo;
        		campos[5]=PanelControles.this.PrimaVacacional;
        		campos[6]=PanelControles.this.Colegiatura;
        		campos[7]=PanelControles.this.SGMM;
        		campos[8]=PanelControles.this.GastosFunerarios;
        		campos[9]=PanelControles.this.MedicinaHospitalizacion;
        		campos[10]=PanelControles.this.Hipotecario;
        		campos[11]=PanelControles.this.Donativos;
        		campos[12]=PanelControles.this.SubcuentaRetiro;
        		campos[13]=PanelControles.this.TransporteEscolar;
        		
        		for(int i=3;i<14;i++){
        			if(trabajador[i].equals("")){
        				pers[i]=0.0;
        			}else{
        				try{
        					pers[i]=Double.parseDouble(trabajador[i]);
        				}catch(NumberFormatException p){
        					campos[i].setText("*DATO INVÁLIDO*");
        				}
        			}
        		}
        		persona=new Persona(Nombre,RFC,pers[3],pers[4],pers[5],educacion,
        									pers[6],pers[7],pers[8],pers[9],pers[10],pers[11],
        									pers[12],pers[13]);
        		deduccion=new Deduccion(persona);
        		deduccion.totalPagar();
        		double total=deduccion.totalPagar();
        		pd.Nombre.setText(Nombre);
        		pd.RFC.setText(RFC);
        		pd.SaldoMensual.setText(Double.toString(pers[3]));
        		pd.Aguinaldo.setText(Double.toString(pers[4]));
        		pd.PrimaVacacional.setText(Double.toString(pers[5]));
        		pd.IngresoAnual.setText(Double.toString(persona.ingresoAnual()));
        		pd.AguinaldoExcento.setText(Double.toString(persona.aguinaldoExento()));
        		pd.AguinaldoGravado.setText(Double.toString(persona.aguinaldoGravado()));
        		pd.PrimaVacacionalExcento.setText(Double.toString(persona.primaExenta));
        		pd.PrimaVacacionalGravado.setText(Double.toString(persona.primaGravado()));
        		pd.TotalIngresosGravados.setText(Double.toString(persona.getTIngresosGravan()));
        		pd.DeduccionesPermitidas.setText(Double.toString(deduccion.dPermitida()));
        		pd.MontoCalculadoISR.setText(Double.toString(deduccion.montoISR()));
        		pd.CuotaFija.setText(Double.toString(deduccion.cuotaFija));
        		pd.PorcentajeExcedido.setText(Double.toString(deduccion.porcentaje));
        		pd.PagoExcedido.setText(Double.toString(deduccion.pagoEx));
        		pd.TotalPagar.setText(Double.toString(total));
        		pd.MedicinaHospitalarios.setText(Double.toString(persona.medHosp));
        		pd.GastosFunerarios.setText(Double.toString(persona.funerarios));
        		pd.SGMM.setText(Double.toString(persona.SGM));
        		pd.Hipotecario.setText(Double.toString(persona.hipoteca));
        		pd.Donativos.setText(Double.toString(persona.donativos));
        		pd.SubcuentaRetiro.setText(Double.toString(persona.subRetiro));
        		pd.TransporteEscolar.setText(Double.toString(persona.Transporte));
        		pd.NivelEducativo.setText(persona.educacion);
        		pd.MaximoDeduccionColegiatura.setText(Double.toString(persona.dEducacion));
        		pd.ColegiaturaPagada.setText(Double.toString(persona.colegiatura));
        		pd.TotalDeduccion.setText(Double.toString(persona.getTotalDeducciones()));
        		
        		resultados[0]=pd.Nombre.getText();
        		resultados[1]=pd.RFC.getText();
        		resultados[2]=pd.SaldoMensual.getText();
        		resultados[3]=pd.IngresoAnual.getText();
        		resultados[4]=pd.Aguinaldo.getText();
        		resultados[5]=pd.AguinaldoExcento.getText();
        		resultados[6]=pd.AguinaldoGravado.getText();
        		resultados[7]=pd.PrimaVacacional.getText();
        		resultados[8]=pd.PrimaVacacionalExcento.getText();
        		resultados[9]=pd.PrimaVacacionalGravado.getText();
        		resultados[10]=pd.TotalIngresosGravados.getText();
        		resultados[11]=pd.MedicinaHospitalarios.getText();
        		resultados[12]=pd.GastosFunerarios.getText();
        		resultados[13]=pd.SGMM.getText();
        		resultados[14]=pd.Hipotecario.getText();
        		resultados[15]=pd.Donativos.getText();
        		resultados[16]=pd.SubcuentaRetiro.getText();
        		resultados[17]=pd.TransporteEscolar.getText();
        		resultados[18]=pd.NivelEducativo.getText();
        		resultados[19]=pd.MaximoDeduccionColegiatura.getText();
        		resultados[20]=pd.ColegiaturaPagada.getText();
        		resultados[21]=pd.TotalDeduccion.getText();
        		resultados[22]=pd.DeduccionesPermitidas.getText();
        		resultados[23]=pd.MontoCalculadoISR.getText();
        		resultados[24]=pd.CuotaFija.getText();
        		resultados[25]=pd.PorcentajeExcedido.getText();
        		resultados[26]=pd.PagoExcedido.getText();
        		resultados[27]=pd.TotalPagar.getText();
        		
     
        		
        		JOptionPane.showMessageDialog(null,total);
        	}
        });panelRight.add(btCalcular);
  
       
        panelRight.add(new JLabel(" "));
        btLimpiar.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                PanelControles.this.Nombre.setText(" ");
                PanelControles.this.RFC.setText(" ");
                PanelControles.this.SaldoMensual.setText(" ");
                PanelControles.this.Aguinaldo.setText(" ");
                PanelControles.this.PrimaVacacional.setText(" ");
                PanelControles.this.MedicinaHospitalizacion.setText(" ");
                PanelControles.this.GastosFunerarios.setText(" ");
                PanelControles.this.SGMM.setText(" ");
                PanelControles.this.Hipotecario.setText(" ");
                PanelControles.this.Donativos.setText(" ");
                PanelControles.this.SubcuentaRetiro.setText(" ");
                PanelControles.this.TransporteEscolar.setText(" ");
                PanelControles.this.Colegiatura.setText(" ");
                rbNinguno.setSelected(true);
            }
        });
        panelRight.add(btLimpiar);
        PanelControles.this.add(panelLeft,BorderLayout.WEST);
        PanelControles.this.add(panelRight,BorderLayout.EAST);
        
        btExportar.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
            	FileNameExtensionFilter filt=new FileNameExtensionFilter("CSV File","csv");
            	fc.setFileFilter(filt);
            	fc.showSaveDialog(null);
            	exportar();
            }
        });this.pd.panelRightLeft.add(btExportar);
      
    }
    public Persona getPersona(){
    	return this.persona;
    }
    public String toString(){
    	String aux="";
 
    	for(int i=0;i<this.resultados.length;i++){
    		if(i==this.resultados.length){
    			aux+=this.resultados[i];
    		}else{
    			aux+=this.resultados[i]+",";
    		}
    	}
    	return aux;
    }
    public void exportar(){
    	File f=this.fc.getSelectedFile();
    	try {
			PrintWriter pw=new PrintWriter(f+".csv");
			pw.println(this.setTitulo());
			pw.println(this.toString());
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    public String setTitulo() {
    	return this.titulo=("Nombre"+","+"RFC"+","+"Sueldo Mensual"+","+"Ingreso Anual"+","+"Aguinaldo"+","
				+ "Aguinaldo Excento"+","+"Aguinaldo Gravado"+","+"Prima Vacacional"+","+"Vacacional Excento"
				+","+"Vacacional Gravada"+","+"Ingresos Gravados"+","+"Medicos"+","+"Funerarios"+","+"SGMM"
				+","+"Hipoteca"+","+"Donativos"+","+"Subcuenta"+","+"Transporte Escolar"+","+"Nivel Educativo"+
				","+"Max Deduccion Colegiatura"+","+"Colegiatura Pagada"+","+"Total Deduccion"+","+
				"Deducciones Permitidas"+","+"Monto ISR"+","+"Cuota Fija"+","+"Porcentaje Excedido"+","+"Pago Excedido"+","+"Total Pagar");
    }
}
